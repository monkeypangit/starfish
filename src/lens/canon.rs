//! Reading Canon's lens correction tables out of CR2 and CR3 files.
//!
//! Both keep them in the Canon maker note as tag 0x4015 (ExifTool's
//! `VignettingCorr`) and, as four-byte words, the settings the camera used
//! for its JPEG as tag 0x4016. A CR2 has them in the ordinary maker note. A
//! CR3 keeps only a reduced maker note in its `CMT3` box and the full tables
//! in the Canon timed-metadata ("CTMD") track: the first sample holds records
//! `[u32 size, u16 type, 6 more bytes, payload]`, and the payload of type 9 is
//! `[u32 size, u32 tag]` and a TIFF with its own maker note.
//!
//! Tag 0x4015 is an array of `u16` laid out as follows (every table is `n`
//! entries long; a table that is not used is all zeros):
//!
//! - word 0 version, word 1 the size in bytes, then settings words
//! - three words `pixel pitch in nm, width, height` of the image
//! - for CR3, `n n n` and then the vignetting table: about `8191` is 100% and the
//!   values fall toward the edge. A CR2 has the same table straight after
//!   the image size, without the count
//! - the knots of the tables, `n - 1` increasing radii (the first, at 0, is
//!   not stored). A radius is in units of 6 µm on the sensor, so a pixel is
//!   `pitch / 6000` of a unit
//! - tables of `n` values starting at `16384` (1.0): two near 1 whose use is
//!   unknown, then the radial ratio of distortion correction (CR3 only)
//!
//! More tables follow (a coarser second set and a copy of the distortion
//! table); they are not read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// What the file says about its correction, still as numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Tables {
    pub pitch_nm: f64,
    pub width: u32,
    pub height: u32,
    /// Vignetting `8191 / gain` at the knots.
    pub vignetting: Vec<f64>,
    /// Knot radii, in units of 6 µm; the first is 0.
    pub knots: Vec<f64>,
    /// Source radius over output radius, times 16384, at the knots.
    pub distortion: Option<Vec<f64>>,
    /// The camera's settings for its JPEG (`None` if the tag is missing).
    pub applied: Option<Applied>,
}

/// Which corrections the camera applied to its JPEG: words 5 (peripheral
/// lighting) and 7 (distortion) of tag 0x4016, 1 meaning on. The words fit
/// every one of a dozen JPEGs inspected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    pub vignetting: bool,
    pub distortion: bool,
}

/// The tables of the Canon file at `path`, if it is one that has them.
pub fn read(path: &Path) -> Option<Tables> {
    let mut file = File::open(path).ok()?;
    let mut head = [0; 12];
    file.read_exact(&mut head).ok()?;
    let (table, flags) = if &head[4..8] == b"ftyp" {
        cr3(&mut file)?
    } else if &head[..4] == b"II*\0" && &head[8..10] == b"CR" {
        cr2(&mut file)?
    } else {
        return None;
    };
    let mut tables = parse_tables(&table)?;
    tables.applied = flags.and_then(|w| {
        (w.len() > 7).then(|| Applied {
            vignetting: w[5] != 0,
            distortion: w[7] != 0,
        })
    });
    Some(tables)
}

/// Parses the `u16` array of tag 0x4015 (see the module docs).
pub fn parse_tables(bytes: &[u8]) -> Option<Tables> {
    let u: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    if u.len() < 40 {
        return None;
    }
    // The first value near 100% starts the vignetting table.
    let start = (16..u.len() - 1).find(|&i| (8100..=8300).contains(&u[i]) && u[i + 1] < u[i])?;
    let n = if u[start - 1] == u[start - 2] && u[start - 2] == u[start - 3] && (8..40).contains(&u[start - 1]) {
        u[start - 1] as usize
    } else {
        u[start..].iter().take_while(|&&v| v != 0).count()
    };
    if !(4..40).contains(&n) || start + n > u.len() {
        return None;
    }
    let vignetting: Vec<f64> = u[start..start + n].iter().map(|&v| v as f64).collect();
    // The curve may wobble a little (the RF 85mm macro's rises by 0.6%) and
    // fall far past the corner (771 of 8191 at the last knot of an RF 24-50
    // at 35mm), but zero or a value above 100% is not a vignetting curve.
    if vignetting.iter().any(|&v| !(1.0..=8300.0).contains(&v)) {
        return None;
    }

    // Pixel pitch, width and height: three words before the tables.
    let (pitch_nm, width, height) = (8..start.min(40)).find_map(|k| {
        let (p, w, h) = (u[k], u[k + 1], u[k + 2]);
        ((2000..9000).contains(&p) && (3000..12000).contains(&w) && (2000..9000).contains(&h) && w > h)
            .then_some((p as f64, w as u32, h as u32))
    })?;

    // Runs of non-zero words after the vignetting table.
    let mut runs: Vec<&[u16]> = Vec::new();
    let mut i = start + n;
    while i < u.len() {
        if u[i] == 0 {
            i += 1;
            continue;
        }
        let j = (i..u.len()).find(|&j| u[j] == 0).unwrap_or(u.len());
        runs.push(&u[i..j]);
        i = j;
    }
    // A CR3 (which has the count) separates its tables with zeros. A CR2
    // does not: its knots follow the vignetting table after one zero.
    let counted = n == u[start - 1] as usize && u[start - 1] == u[start - 2];
    let knots: &[u16] = if counted {
        runs.iter()
            .find(|r| r.len() == n - 1 && r.windows(2).all(|w| w[0] < w[1]))?
    } else {
        u.get(start + n + 1..start + 2 * n)?
    };
    if !knots.windows(2).all(|w| w[0] < w[1]) || knots.first().is_none_or(|&k| k == 0) {
        return None;
    }
    let knots: Vec<f64> = std::iter::once(0.0).chain(knots.iter().map(|&k| k as f64)).collect();
    let tables: Vec<&&[u16]> = runs.iter().filter(|r| r.len() == n && r[0] == 16384).collect();
    let distortion = (counted && tables.len() >= 3).then(|| tables[2].iter().map(|&v| v as f64).collect());
    Some(Tables {
        pitch_nm,
        width,
        height,
        vignetting,
        knots,
        distortion,
        applied: None,
    })
}

/// A little-endian TIFF held in memory; offsets are from the start of `buf`.
struct Mem<'a> {
    buf: &'a [u8],
}

struct MemEntry<'a> {
    tag: u16,
    data: &'a [u8],
}

impl<'a> Mem<'a> {
    fn u32(&self, at: usize) -> Option<u32> {
        Some(u32::from_le_bytes(self.buf.get(at..at + 4)?.try_into().ok()?))
    }

    fn entries(&self, ifd: usize) -> Option<Vec<MemEntry<'a>>> {
        let count = u16::from_le_bytes(self.buf.get(ifd..ifd + 2)?.try_into().ok()?) as usize;
        if count > 512 {
            return None;
        }
        (0..count)
            .map(|i| {
                let at = ifd + 2 + 12 * i;
                let e = self.buf.get(at..at + 12)?;
                let tag = u16::from_le_bytes([e[0], e[1]]);
                let kind = u16::from_le_bytes([e[2], e[3]]);
                let n = u32::from_le_bytes(e[4..8].try_into().ok()?) as usize;
                let size = n.checked_mul(match kind {
                    3 | 8 => 2,
                    4 | 9 | 11 | 13 => 4,
                    5 | 10 | 12 => 8,
                    _ => 1,
                })?;
                let offset = if size <= 4 { at + 8 } else { self.u32(at + 8)? as usize };
                Some(MemEntry {
                    tag,
                    data: self.buf.get(offset..offset.checked_add(size)?)?,
                })
            })
            .collect()
    }
}

fn words(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Tag 0x4015 and the words of 0x4016 from a Canon maker note IFD at
/// `ifd` in `mem`.
fn maker_note(mem: &Mem, ifd: usize) -> Option<(Vec<u8>, Option<Vec<u32>>)> {
    let entries = mem.entries(ifd)?;
    let table = entries.iter().find(|e| e.tag == 0x4015)?.data.to_vec();
    let flags = entries.iter().find(|e| e.tag == 0x4016).map(|e| words(e.data));
    Some((table, flags))
}

/// An entry of an IFD read from a file: where its value is and how long.
struct FileEntry {
    tag: u16,
    offset: u64,
    size: usize,
}

/// The entries of the IFD at `ifd` in a little-endian TIFF file whose
/// offsets are from the start of the file.
fn file_entries(file: &mut File, ifd: u64) -> Option<Vec<FileEntry>> {
    let mut count = [0; 2];
    file.seek(SeekFrom::Start(ifd)).ok()?;
    file.read_exact(&mut count).ok()?;
    let count = u16::from_le_bytes(count) as usize;
    if count > 512 {
        return None;
    }
    let mut raw = vec![0; count * 12];
    file.read_exact(&mut raw).ok()?;
    raw.chunks_exact(12)
        .enumerate()
        .map(|(i, e)| {
            let kind = u16::from_le_bytes([e[2], e[3]]);
            let n = u32::from_le_bytes(e[4..8].try_into().ok()?) as usize;
            let size = n.checked_mul(match kind {
                3 | 8 => 2,
                4 | 9 | 11 | 13 => 4,
                5 | 10 | 12 => 8,
                _ => 1,
            })?;
            let at = ifd + 2 + 12 * i as u64;
            let offset = if size <= 4 {
                at + 8
            } else {
                u32::from_le_bytes(e[8..12].try_into().ok()?) as u64
            };
            Some(FileEntry {
                tag: u16::from_le_bytes([e[0], e[1]]),
                offset,
                size,
            })
        })
        .collect()
}

fn read_at(file: &mut File, offset: u64, size: usize) -> Option<Vec<u8>> {
    if size > 4 << 20 {
        return None;
    }
    let mut buf = vec![0; size];
    file.seek(SeekFrom::Start(offset)).ok()?;
    file.read_exact(&mut buf).ok()?;
    Some(buf)
}

/// CR2: the maker note is in the Exif IFD, offsets from the file start. The
/// IFDs can be anywhere, even at the end of the file, so they are read in
/// place.
fn cr2(file: &mut File) -> Option<(Vec<u8>, Option<Vec<u32>>)> {
    let ifd0 = u32::from_le_bytes(read_at(file, 4, 4)?.try_into().ok()?) as u64;
    let root = file_entries(file, ifd0)?;
    let exif = root.iter().find(|e| e.tag == 0x8769)?;
    let exif = u32::from_le_bytes(read_at(file, exif.offset, 4)?.try_into().ok()?) as u64;
    let exif = file_entries(file, exif)?;
    let note = exif.iter().find(|e| e.tag == 0x927c)?;
    let entries = file_entries(file, note.offset)?;
    let table = entries.iter().find(|e| e.tag == 0x4015)?;
    let table = read_at(file, table.offset, table.size)?;
    let flags = entries
        .iter()
        .find(|e| e.tag == 0x4016)
        .and_then(|e| read_at(file, e.offset, e.size))
        .map(|d| words(&d));
    Some((table, flags))
}

/// The boxes in `buf` as (type, payload).
fn boxes(buf: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = Vec::new();
    let mut p = 0;
    while p + 8 <= buf.len() {
        let size = u32::from_be_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as usize;
        let kind = [buf[p + 4], buf[p + 5], buf[p + 6], buf[p + 7]];
        let (header, size) = match size {
            0 => (8, buf.len() - p),
            1 => match buf.get(p + 8..p + 16) {
                Some(b) => (16, u64::from_be_bytes(b.try_into().unwrap()) as usize),
                None => break,
            },
            n => (8, n),
        };
        if size < header || p + size > buf.len() {
            break;
        }
        out.push((kind, &buf[p + header..p + size]));
        p += size;
    }
    out
}

fn child<'a>(buf: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(buf).into_iter().find(|(k, _)| k == kind).map(|(_, b)| b)
}

/// CR3: the tables from CTMD record 9, the settings from the `CMT3` box.
fn cr3(file: &mut File) -> Option<(Vec<u8>, Option<Vec<u32>>)> {
    let length = file.metadata().ok()?.len();
    // The movie box, among the top-level boxes.
    let mut at = 0u64;
    let mut moov = Vec::new();
    while at + 8 <= length {
        let mut head = [0; 16];
        file.seek(SeekFrom::Start(at)).ok()?;
        let got = file.read(&mut head).ok()?;
        if got < 8 {
            return None;
        }
        let size = u32::from_be_bytes(head[..4].try_into().ok()?) as u64;
        let (header, size) = match size {
            0 => (8, length - at),
            1 => (16, u64::from_be_bytes(head[8..16].try_into().ok()?)),
            n => (8, n),
        };
        if size < header {
            return None;
        }
        if &head[4..8] == b"moov" {
            if size - header > 64 << 20 {
                return None;
            }
            file.seek(SeekFrom::Start(at + header)).ok()?;
            moov = vec![0; (size - header) as usize];
            file.read_exact(&mut moov).ok()?;
            break;
        }
        at += size;
    }

    // The settings: the CMT3 box inside the uuid box is a whole TIFF.
    let flags = boxes(&moov)
        .into_iter()
        .filter(|(k, _)| k == b"uuid")
        .filter_map(|(_, b)| b.get(16..))
        .find_map(|b| child(b, b"CMT3"))
        .and_then(|tiff| {
            let mem = Mem { buf: tiff };
            let entries = mem.entries(mem.u32(4)? as usize)?;
            Some(words(entries.iter().find(|e| e.tag == 0x4016)?.data))
        });

    // The timed metadata track: the first sample of the CTMD track.
    for (kind, trak) in boxes(&moov) {
        if &kind != b"trak" {
            continue;
        }
        let stbl = child(trak, b"mdia")
            .and_then(|b| child(b, b"minf"))
            .and_then(|b| child(b, b"stbl"));
        let Some(stbl) = stbl else { continue };
        if !child(stbl, b"stsd").is_some_and(|b| b.windows(4).any(|w| w == b"CTMD")) {
            continue;
        }
        let stsz = child(stbl, b"stsz")?;
        let sample_size = u32::from_be_bytes(stsz.get(4..8)?.try_into().ok()?);
        let size = if sample_size != 0 {
            sample_size
        } else {
            u32::from_be_bytes(stsz.get(12..16)?.try_into().ok()?)
        } as usize;
        let offset = match child(stbl, b"stco") {
            Some(b) => u32::from_be_bytes(b.get(8..12)?.try_into().ok()?) as u64,
            None => u64::from_be_bytes(child(stbl, b"co64")?.get(8..16)?.try_into().ok()?),
        };
        if size > 16 << 20 {
            return None;
        }
        let mut sample = vec![0; size];
        file.seek(SeekFrom::Start(offset)).ok()?;
        file.read_exact(&mut sample).ok()?;

        let mut p = 0;
        while p + 12 <= sample.len() {
            let record = u32::from_le_bytes(sample[p..p + 4].try_into().ok()?) as usize;
            let kind = u16::from_le_bytes([sample[p + 4], sample[p + 5]]);
            if record < 12 || p + record > sample.len() {
                break;
            }
            if kind == 9 {
                let note = sample.get(p + 12 + 8..p + record)?;
                let mem = Mem { buf: note };
                let (table, _) = maker_note(&mem, mem.u32(4)? as usize)?;
                return Some((table, flags));
            }
            p += record;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes of a tag 0x4015 of `len` words with the given runs of
    /// non-zero words at their positions (everything else zero), which is
    /// all the parser reads of a real one.
    fn bytes((len, runs): (usize, &[(usize, &[u16])])) -> Vec<u8> {
        let mut words = vec![0u16; len];
        for (at, values) in runs {
            words[*at..*at + values.len()].copy_from_slice(values);
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// RF-S 18-45mm at 18mm on an R50 (n = 22): distortion applied by the camera.
    const R50_18MM: (usize, &[(usize, &[u16])]) = (750, &[
        (0, &[12544, 1500, 1, 1, 1, 1, 1, 1]),
        (10, &[70, 100, 100, 100, 1740, 128, 1740, 128, 3720, 6000, 4000]),
        (66, &[22, 22, 22, 8191, 8179, 8140, 8075, 7985, 7867, 7723, 7555, 7363, 7134, 6695, 6361, 6002, 5606, 5198, 4791, 4382, 3975, 3566, 3159, 2750, 2342]),
        (185, &[189, 379, 568, 757, 946, 1136, 1325, 1514, 1703, 1893, 1987, 2082, 2177, 2271, 2366, 2461, 2555, 2650, 2745, 2839, 2934]),
        (299, &[16384, 16437, 16490, 16536, 16577, 16606, 16627, 16640, 16635, 16621, 16590, 16566, 16535, 16495, 16453, 16411, 16365, 16328, 16291, 16250, 16215, 16193]),
        (322, &[16384, 16369, 16354, 16349, 16353, 16367, 16395, 16439, 16494, 16563, 16653, 16710, 16769, 16836, 16916, 17001, 17095, 17191, 17289, 17403, 17501, 17574]),
        (346, &[189, 379, 568, 757, 946, 1136, 1325, 1514, 1703, 1893, 1987, 2082, 2177, 2271, 2366, 2461, 2555, 2650, 2745, 2839, 2934]),
        (368, &[16384, 16311, 16266, 16192, 16087, 15956, 15802, 15627, 15438, 15238, 15029, 14922, 14815, 14706, 14598, 14489, 14380, 14273, 14166, 14059, 13957, 13861]),
        (392, &[189, 379, 568, 757, 946, 1136, 1325, 1514, 1703, 1893, 1987, 2082, 2177, 2271, 2366, 2461, 2555, 2650, 2745, 2839, 2934]),
        (419, &[17, 16384, 16418, 16466, 16518, 16562, 16592, 16603, 16594, 16563, 16516, 16456, 16393, 16336, 16298, 16296, 16347, 16474]),
        (443, &[16384, 16388, 16374, 16357, 16350, 16361, 16397, 16458, 16543, 16647, 16761, 16873, 16968, 17027, 17027, 16977, 17150]),
        (467, &[183, 366, 550, 733, 916, 1100, 1283, 1467, 1650, 1833, 2017, 2200, 2383, 2567, 2750, 2934]),
    ]);
    /// EF-M 15-45mm at 23mm on an M50 (n = 15).
    const M50_23MM: (usize, &[(usize, &[u16])]) = (316, &[
        (0, &[12320, 632, 1]),
        (4, &[1, 1, 1]),
        (8, &[70]),
        (10, &[100, 100, 1740, 128]),
        (16, &[3720, 6000, 4000]),
        (64, &[15, 15, 15, 8191, 8172, 8114, 8020, 7892, 7732, 7544, 7328, 7087, 6775, 6474, 6061, 5618, 5117, 4766]),
        (91, &[189, 379, 568, 757, 946, 1136, 1325, 1514, 1703, 1893, 2082, 2271, 2461, 2586]),
        (113, &[16384, 16411, 16433, 16457, 16482, 16507, 16528, 16546, 16563, 16577, 16590, 16596, 16599, 16588, 16578]),
        (136, &[16384, 16379, 16371, 16367, 16365, 16368, 16372, 16382, 16393, 16415, 16444, 16483, 16541, 16609, 16660]),
        (160, &[189, 379, 568, 757, 946, 1136, 1325, 1514, 1703, 1893, 2082, 2271, 2461, 2586]),
        (182, &[16384, 16336, 16332, 16325, 16316, 16304, 16292, 16280, 16272, 16268, 16271, 16283, 16308, 16347, 16383]),
        (206, &[189, 379, 568, 757, 946, 1136, 1325, 1514, 1703, 1893, 2082, 2271, 2461, 2586]),
        (301, &[1, 23]),
        (304, &[1, 1]),
        (308, &[225]),
    ]);
    /// EF 35mm f/2 IS on a 5D Mark IV, a CR2 (n = 6, no counts, no distortion).
    const EOS_5D4_35MM: (usize, &[(usize, &[u16])]) = (306, &[
        (0, &[12288, 612]),
        (12, &[1228, 128]),
        (15, &[128, 5360, 6720, 4480, 8191, 6892, 4928, 3237, 2340, 1895]),
        (26, &[1136, 2271, 3218, 3786, 4095, 16384, 16462, 16533, 16461, 16315, 16192, 16384, 16364, 16361, 16550, 16811, 17014]),
        (44, &[1136, 2271, 3218, 3786, 4095, 16276, 16233, 16109, 15988, 15923, 15894]),
        (56, &[1136, 2271, 3218, 3786, 4095]),
        (231, &[15]),
        (233, &[76]),
        (235, &[77]),
        (237, &[75]),
        (239, &[73]),
        (241, &[72]),
        (243, &[70]),
        (245, &[70]),
        (247, &[63]),
        (249, &[56]),
        (251, &[54]),
        (253, &[33]),
        (255, &[1]),
        (265, &[51]),
        (267, &[99]),
        (269, &[86]),
        (271, &[93]),
        (273, &[95]),
        (275, &[89]),
        (277, &[73]),
        (279, &[77]),
        (281, &[44]),
        (283, &[51]),
        (285, &[56]),
        (287, &[59]),
        (289, &[59]),
        (291, &[62]),
        (293, &[65]),
        (295, &[72]),
        (297, &[67]),
        (299, &[64]),
        (302, &[35]),
        (304, &[1]),
    ]);

    #[test]
    fn a_cr3_table_is_read_with_its_distortion() {
        let t = parse_tables(&bytes(R50_18MM)).unwrap();
        assert_eq!((t.pitch_nm, t.width, t.height), (3720.0, 6000, 4000));
        assert_eq!(t.vignetting.len(), 22);
        assert_eq!((t.vignetting[0], t.vignetting[21]), (8191.0, 2342.0));
        assert_eq!(t.knots.len(), 22);
        assert_eq!((t.knots[0], t.knots[1], t.knots[21]), (0.0, 189.0, 2934.0));
        // The third table, the one that falls: the first two stay near 1.
        let d = t.distortion.unwrap();
        assert_eq!((d[0], d[21]), (16384.0, 13861.0));
    }

    #[test]
    fn a_cr3_without_distortion_in_use_still_has_its_third_table() {
        // The camera left this one (an EF-M lens) uncorrected, but a table
        // of small wobbles is there.
        let t = parse_tables(&bytes(M50_23MM)).unwrap();
        assert_eq!(t.vignetting.len(), 15);
        assert_eq!(*t.knots.last().unwrap(), 2586.0);
        let d = t.distortion.unwrap();
        assert!(d.iter().all(|&v| (v / 16384.0 - 1.0).abs() < 0.01));
    }

    #[test]
    fn a_cr2_has_no_counts_and_no_distortion() {
        let t = parse_tables(&bytes(EOS_5D4_35MM)).unwrap();
        assert_eq!((t.pitch_nm, t.width, t.height), (5360.0, 6720, 4480));
        assert_eq!(t.vignetting, [8191.0, 6892.0, 4928.0, 3237.0, 2340.0, 1895.0]);
        assert_eq!(t.knots, [0.0, 1136.0, 2271.0, 3218.0, 3786.0, 4095.0]);
        assert!(t.distortion.is_none());
    }

    #[test]
    fn a_vignetting_table_may_fall_far_past_the_corner() {
        // An RF 24-50 at 35mm ends at 771 of 8191, a gain over 10 beyond the
        // corner of the frame.
        let mut far = bytes(R50_18MM);
        far[2 * 90..2 * 90 + 2].copy_from_slice(&771u16.to_le_bytes());
        assert_eq!(parse_tables(&far).unwrap().vignetting[21], 771.0);
    }

    #[test]
    fn nonsense_is_not_a_table() {
        assert!(parse_tables(&[]).is_none());
        assert!(parse_tables(&[0; 600]).is_none());
        let mut bad = bytes(R50_18MM);
        bad[2 * 69] = 0xff; // a vignetting value of 65535
        bad[2 * 69 + 1] = 0xff;
        assert!(parse_tables(&bad).is_none());
    }
}
