//! Right-pane histogram of the developed image: R, G and B counts over the
//! 8-bit display values (plus their combined luminance as a grey line) of a
//! subset of the export's pixels, drawn on the GPU by the preview's own
//! pipeline (`develop.wgsl`) and counted there (`histogram.wgsl`), so it
//! covers exactly what the export holds: crop, lens corrections and local
//! contrast included.

use std::sync::Mutex;

use iced::advanced::widget::{Tree, Widget};
use iced::advanced::{Clipboard, Layout, Shell, layout, renderer};
use iced::widget::Action;
use iced::widget::canvas::{self, Frame, Geometry, Path, Stroke};
use iced::{Color, Element, Event, Length, Point, Rectangle, Renderer, Size, Theme, mouse};

use crate::agx::{self, Curve};
use crate::export;
use crate::preview::{self, MID_GRAY, PreviewPipeline, PrimitiveImage};
use crate::processing::Image;

const BINS: usize = 256;
/// Enough for a smooth histogram while recomputing on every slider step.
const MAX_SAMPLES: u32 = 1 << 17;
/// Longest side of the draw the bins are counted from.
const MAX_SIDE: u32 = 4096;
/// The draw's format: f32, as the export's.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

#[derive(Debug, Clone)]
pub struct Histogram {
    /// R, G, B, then luminance.
    bins: [[u32; BINS]; 4],
}

impl Histogram {
    /// The histogram of what the export of `image` drawn as `view` would
    /// hold, from every few of its pixels (at most about [`MAX_SAMPLES`]),
    /// each exactly the export's pixel. It is drawn by the preview's own
    /// pipeline on iced's device and counted there. `None` until the
    /// preview has been drawn (and so iced's device is known), after a
    /// short wait so a caller can simply try again.
    pub fn compute(image: &Image, view: export::View) -> Result<Option<Self>, String> {
        let Some((device, queue)) = preview::gpu() else {
            std::thread::sleep(std::time::Duration::from_millis(50));
            return Ok(None);
        };
        Self::compute_on(&device, &queue, image, view).map(Some)
    }

    /// [`Histogram::compute`] on `device`.
    pub fn compute_on(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Image,
        view: export::View,
    ) -> Result<Self, String> {
        static PASS: Mutex<Option<Pass>> = Mutex::new(None);
        let mut pass = PASS.lock().unwrap_or_else(|e| e.into_inner());
        if pass.as_ref().is_none_or(|p| p.device != *device) {
            *pass = Some(Pass::new(device));
        }
        let pass = pass.as_mut().unwrap();
        // As export: an error of iced's in the meantime lands here too.
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bins = pass.run(device, queue, image, view);
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            return Err(format!("GPU error in the histogram: {error}"));
        }
        Ok(Self { bins: bins? })
    }
}

/// The histogram's GPU objects, kept between runs.
struct Pass {
    device: wgpu::Device,
    draw: PreviewPipeline,
    count: wgpu::ComputePipeline,
    bins: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl Pass {
    fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("shaders/histogram.wgsl"));
        let size = (4 * BINS * std::mem::size_of::<u32>()) as u64;
        Self {
            device: device.clone(),
            draw: PreviewPipeline::with_format(device, FORMAT),
            count: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("histogram"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bins: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("histogram bins"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("histogram readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Image,
        view: export::View,
    ) -> Result<[[u32; BINS]; 4], String> {
        // Every `stride`-th output pixel each way: target pixel i is output
        // pixel i · stride + stride / 2, its centre drawn exactly there (with
        // an even stride, (i + 0.5) · stride would fall on a pixel edge).
        let (width, height) = export::frame_pixels(image, view.rotation, view.crop);
        let stride = ((width as f32 * height as f32 / MAX_SAMPLES as f32).sqrt().ceil() as u32)
            .max(width.div_ceil(MAX_SIDE))
            .max(height.div_ceil(MAX_SIDE))
            .max(1);
        let (w, h) = (width.div_ceil(stride), height.div_ceil(stride));
        let shot = PrimitiveImage {
            image: image.clone(),
            scale: 1.0 / stride as f32,
            center: {
                let offset = (stride / 2) as f32 + 0.5 - stride as f32 / 2.0;
                Point::new((w * stride) as f32 / 2.0 + offset, (h * stride) as f32 / 2.0 + offset)
            },
            adjustments: view.adjustments,
            rotation: view.rotation,
            crop: view.crop,
            cropping: false,
            lens: view.lens,
            point: true,
        };
        self.draw.prepare_image(device, queue, &shot, [w as f32, h as f32])?;

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("histogram"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("histogram"),
            layout: &self.count.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.bins.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("histogram draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Alpha 0: not counted (past the frame).
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            self.draw.draw_image(&mut pass);
        }
        encoder.clear_buffer(&self.bins, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.count);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(w.div_ceil(16), h.div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(&self.bins, 0, &self.readback, 0, None);
        queue.submit([encoder.finish()]);
        // Not kept between runs: it would hold an image the app has moved on
        // from while the next one uploads (the submission keeps it till done).
        self.draw.release();

        let slice = self.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| format!("GPU error: {e}"))?;
        receiver
            .recv()
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("Failed to read the histogram back: {e}"))?;
        let mut bins = [[0; BINS]; 4];
        {
            let data = slice.get_mapped_range();
            let counts: &[u32] = bytemuck::cast_slice(&data);
            for (channel, bins) in bins.iter_mut().enumerate() {
                bins.copy_from_slice(&counts[channel * BINS..][..BINS]);
            }
        }
        self.readback.unmap();
        Ok(bins)
    }
}

fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_encode(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Draws the histogram (if an image is loaded) and, over it, the tone curve:
/// curve input left to right, output bottom to top, both sRGB-encoded 0..1
/// like the histogram's bins (the curve itself works in `agx`'s power-2.4
/// encoding, converted here), with the diagonal for reference and a dot where
/// they cross. For grey pixels with neutral saturation and grading the
/// output axis is exactly the histogram's.
pub struct View<'a, Message> {
    pub histogram: Option<&'a Histogram>,
    pub curve: Curve,
    /// Gets the wanted height as a fraction of the width while the lower
    /// border is dragged.
    pub on_resize: fn(f32) -> Message,
}

/// Grabbable height of the lower border.
const GRAB: f32 = 8.0;

impl<Message> canvas::Program<Message> for View<'_, Message> {
    /// Whether the lower border is being dragged.
    type State = bool;

    fn update(
        &self,
        dragging: &mut bool,
        event: &Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let on_border = cursor
                    .position_in(bounds)
                    .is_some_and(|p| p.y >= bounds.height - GRAB);
                *dragging = on_border;
                on_border.then(Action::capture)
            }
            Event::Mouse(mouse::Event::CursorMoved { position }) if *dragging => Some(
                Action::publish((self.on_resize)((position.y - bounds.y) / bounds.width))
                    .and_capture(),
            ),
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) if *dragging => {
                *dragging = false;
                Some(Action::capture())
            }
            _ => None,
        }
    }

    fn mouse_interaction(
        &self,
        dragging: &bool,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        let on_border = cursor
            .position_in(bounds)
            .is_some_and(|p| p.y >= bounds.height - GRAB);
        if *dragging || on_border {
            mouse::Interaction::ResizingVertically
        } else {
            mouse::Interaction::None
        }
    }

    fn draw(
        &self,
        _state: &bool,
        renderer: &Renderer,
        theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let (width, height) = (bounds.width, bounds.height);
        let text = theme.palette().text;

        frame.fill_rectangle(Point::ORIGIN, bounds.size(), Color { a: 0.05, ..text });
        frame.fill_rectangle(
            Point::new(0.0, height - 1.0),
            Size::new(width, 1.0),
            Color { a: 0.15, ..text },
        );
        // Where 18% grey lands on screen.
        let mid = srgb_encode(MID_GRAY) * width;
        frame.fill_rectangle(
            Point::new(mid, 0.0),
            Size::new(1.0, height),
            Color { a: 0.1, ..text },
        );

        if let Some(histogram) = self.histogram {
            draw_bins(&mut frame, histogram);
        }
        self.draw_curve(&mut frame, text);
        vec![frame.into_geometry()]
    }
}

impl<Message> View<'_, Message> {
    fn draw_curve(&self, frame: &mut Frame, color: Color) {
        let (width, height) = (frame.width(), frame.height());
        let at = |x: f32, y: f32| Point::new(x * width, (1.0 - y) * height);
        // sRGB-encoded input → curve → sRGB-encoded output.
        let plot = |x: f32| {
            let y = self.curve.apply(agx::encode(srgb_decode(x)));
            srgb_encode(agx::decode(y))
        };

        let faint = Stroke::default()
            .with_width(1.0)
            .with_color(Color { a: 0.15, ..color });
        frame.stroke(&Path::line(at(0.0, 0.0), at(1.0, 1.0)), faint);

        let steps = width.ceil().max(1.0) as usize;
        let curve = Path::new(|builder| {
            builder.move_to(at(0.0, plot(0.0)));
            for i in 1..=steps {
                let x = i as f32 / steps as f32;
                builder.line_to(at(x, plot(x)));
            }
        });
        let line = Stroke::default()
            .with_width(1.5)
            .with_color(Color { a: 0.8, ..color });
        frame.stroke(&curve, line);

        let pivot = srgb_encode(agx::decode(self.curve.pivot_x));
        let crossover = at(pivot, pivot);
        frame.fill(&Path::circle(crossover, 3.0), Color { a: 0.9, ..color });
    }
}

fn draw_bins(frame: &mut Frame, histogram: &Histogram) {
    let (width, height) = (frame.width(), frame.height());
    // Scale to the tallest interior bin so clipped blacks and whites
    // don't flatten the rest; their spikes are cut off at the top.
    let peak = histogram
        .bins
        .iter()
        .flat_map(|bins| bins[1..BINS - 1].iter())
        .copied()
        .max()
        .unwrap_or(0)
        .max(1) as f32;

    let colors = [
        Color::from_rgba(1.0, 0.27, 0.23, 0.5),
        Color::from_rgba(0.19, 0.82, 0.32, 0.5),
        Color::from_rgba(0.04, 0.52, 1.0, 0.5),
    ];
    for (i, bins) in histogram.bins.iter().enumerate() {
        let path = Path::new(|builder| {
            let filled = i < colors.len();
            if filled {
                builder.move_to(Point::new(0.0, height));
            }
            for (k, &count) in bins.iter().enumerate() {
                let p = Point::new(
                    k as f32 / (BINS - 1) as f32 * width,
                    height * (1.0 - (count as f32 / peak).min(1.0)),
                );
                if filled || k > 0 {
                    builder.line_to(p)
                } else {
                    builder.move_to(p)
                }
            }
            if filled {
                builder.line_to(Point::new(width, height));
                builder.close();
            }
        });
        match colors.get(i) {
            Some(&color) => frame.fill(&path, color),
            None => frame.stroke(
                &path,
                Stroke::default()
                    .with_width(1.5)
                    .with_color(Color::from_rgba(0.8, 0.8, 0.8, 0.9)),
            ),
        }
    }
}

/// Lays out `content` at the full available width and `ratio` times that
/// as height, so it keeps its proportions when the pane is resized.
pub struct Aspect<'a, Message> {
    content: Element<'a, Message>,
    ratio: f32,
}

impl<'a, Message> Aspect<'a, Message> {
    pub fn new(content: impl Into<Element<'a, Message>>, ratio: f32) -> Self {
        Self {
            content: content.into(),
            ratio,
        }
    }
}

impl<Message> Widget<Message, Theme, Renderer> for Aspect<'_, Message> {
    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Shrink)
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_ref(&self.content));
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let width = limits.max().width;
        let size = Size::new(width, width * self.ratio);
        let child = self.content.as_widget_mut().layout(
            &mut tree.children[0],
            renderer,
            &layout::Limits::new(size, size),
        );
        layout::Node::with_children(size, vec![child])
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            style,
            layout.children().next().unwrap(),
            cursor,
            viewport,
        );
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout.children().next().unwrap(),
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout.children().next().unwrap(),
            cursor,
            viewport,
            renderer,
        )
    }
}

impl<'a, Message: 'a> From<Aspect<'a, Message>> for Element<'a, Message> {
    fn from(aspect: Aspect<'a, Message>) -> Self {
        Element::new(aspect)
    }
}
