#!/usr/bin/env python3
# yolo_tkinter_labeler_adaptive.py
#
# Full editor — adapted per user requests:
# - Fixed region zoom
# - Proper alpha blending for overlays (Pillow)
# - Adaptive left panel with tool selection (YOLO, Excessive Green, Manual)
# - Validate all extra predictions -> GT
# - Delete all GT
# - Interactive navigation slider + Prev/Next
#
# Dependencies: Pillow (pip install Pillow)
# Optional: ultralytics for YOLO predictions

import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale, font
from PIL import Image, ImageTk, ImageDraw, ImageStat, ImageFilter

# core imports
from core.utils import yolo_to_pixels, pixels_to_yolo, calculate_iou
from core.models import BBox
from AI.yolo import load_yolo_model, predict_current_image
from MV.eXg import excessive_green_apply, _compute_excessive_green_mask
from MV.Shapes import rectangle_detect, circle_detect, triangle_detect, polygon_detect


# ---------- Modes ----------
MODES = {
    "none": {"label": "None", "color": "#444444"},
    "draw": {"label": "Draw", "color": "#168fd5"},
    "move": {"label": "Move/Resize", "color": "#FFA500"},
    "delete": {"label": "Delete", "color": "#FF5555"},
    "validate": {"label": "Validate", "color": "#28a745"},

}
# "zoom": {"label": "Zoom", "color": "#AAAAFF"}
# ---------- Main application ----------
class YoloEditorApp(tk.Tk):
    HANDLE_SIZE = 6

    def __init__(self):
        super().__init__()
        self.title("YOLO Annotation Editor - Adaptive")
        self.geometry("1200x800")

        # state
        self.image_folder = None
        self.label_folder = None
        self.prediction_folder = None
        self.image_files = []
        self.current_index = -1
        self.current_image = None
        self.tk_image = None

        # scaling / zoom
        self.display_scale = 1.0
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        # when True, keep offset_x/offset_y as set by zoom operations instead of recentering
        self._preserve_offset = False
        self.img_w = 0
        self.img_h = 0

        # annotation data
        self.bboxes_gt = {}     # img_path -> [BBox,...]
        self.bboxes_pred = {}   # img_path -> [BBox,...]
        self.bboxes_extra = {}  # img_path -> [BBox,...]  (if you add extras manually)
        self.selected_bbox = None

        # dragging state
        self.drag_data = {"mode": None, "start": (0, 0), "bbox": None, "handle": None}
        self._drawing_mode = False
        self._new_rect_id = None
        self._new_rect_start = (0, 0)

        # handle mapping
        self.handle_id_to_info = {}

        # yolo model placeholder
        self.yolo_model = None

        # UI vars
        self.current_mode = tk.StringVar(value="none")
        self.mode_buttons = {}

        self.show_mv_preview = tk.BooleanVar(value=True)

        self.show_gt = tk.BooleanVar(value=True)
        self.show_pred = tk.BooleanVar(value=False)  # default unchecked
        self.show_extra = tk.BooleanVar(value=False)

        self.transparency_gt = tk.DoubleVar(value=0.2)
        self.transparency_pred = tk.DoubleVar(value=0.2)
        self.transparency_extra = tk.DoubleVar(value=0.2)

        self.check_keep_ratio = tk.BooleanVar(value=True)

        # Adaptive tool selection
        self.tools = ["YOLO Detection", "Excessive Green"]
        self.selected_tool = tk.StringVar(value=self.tools[0])

        # build UI
        self._build_ui()
    # ---------- UI ----------
    def _build_ui(self):
        bold_font = font.Font(family="TkDefaultFont", size=10, weight="bold")
        # Root layout: left control panel + main viewer
        self.columnconfigure(1, weight=1)

        # ---- LEFT STATIC PANEL ----
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        # Folder section
        folder_label = ttk.Label(left, text="Folder", font=bold_font)
        folder_label.pack(anchor=tk.W, pady=(6, 0))

        ttk.Button(left, text="Select Parent Folder", command=self.select_parent_folder).pack(fill=tk.X, pady=2)
        ttk.Separator(left).pack(fill=tk.X, pady=1)
        # Use a Frame instead of Label
        self.data_panel = tk.Frame(left)
        self.data_panel.pack(fill=tk.X, pady=(2, 2))

        # Section title
        self.data_title = ttk.Label(self.data_panel, text="Image Data", font=bold_font)
        self.data_title.pack(fill=tk.X)

        # One label for both Shape and GT
        # Shape (first row)
        self.lbl_shape_info = ttk.Label(self.data_panel, text="Shape: -", anchor=tk.W)
        self.lbl_shape_info.pack(fill=tk.X)

        # GT / Pred / Extra info (second row)
        self.lbl_counts_info = ttk.Label(self.data_panel, text="GT: 0   Pred: 0   Extra: 0", anchor=tk.W)
        self.lbl_counts_info.pack(fill=tk.X)
        ttk.Separator(left).pack(fill=tk.X, pady=6)


        # Modes
        ttk.Label(left, text="Modes", font=bold_font).pack(anchor=tk.W)
        modes_frame = ttk.Frame(left)
        modes_frame.pack(fill=tk.X, pady=1)
        for mode_key, cfg in MODES.items():
            if mode_key == "none":
                continue
            b = tk.Button(
                modes_frame, text=cfg["label"],
                relief=tk.RAISED, bg=cfg["color"], fg="#000000",
                command=lambda k=mode_key: self.set_mode(k)
            )
            b.pack(fill=tk.X, pady=1)
            self.mode_buttons[mode_key] = b
        ttk.Separator(left).pack(fill=tk.X, pady=1)
        # Visibility sliders
        ttk.Label(left, text="Visibility & Transparency", font=bold_font).pack(anchor=tk.W, pady=(6, 2))
        def make_slider_row(parent, label, var_show, var_transp, color, style_name):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=0)

            # Checkbutton (left)
            cb = ttk.Checkbutton(row, text=label, variable=var_show,
                                command=self._render_image_on_canvas)
            cb.grid(row=0, column=0, sticky="w", padx=(0,6))

            # Slider (right)
            style = ttk.Style()
            style.theme_use("clam")  # ensures color settings are applied
            style.configure(f"{style_name}.Horizontal.TScale", troughcolor="#ffffff")
            style.map(f"{style_name}.Horizontal.TScale",
                    background=[('active', color), ('!active', color)])

            slider = ttk.Scale(row, from_=0, to=1, orient='horizontal', variable=var_transp,
                            length=90, style=f"{style_name}.Horizontal.TScale",
                            command=lambda e: self._render_image_on_canvas())
            slider.grid(row=0, column=1, sticky="e", padx=2)

            # make sure column 1 expands to align sliders to the right
            row.columnconfigure(0, weight=1)
            row.columnconfigure(1, weight=0)  # slider width fixed

        make_slider_row(left, "GT", self.show_gt, self.transparency_gt, "#3cb043", "Green")
        make_slider_row(left, "Pred", self.show_pred, self.transparency_pred, "#ff5050", "Red")
        make_slider_row(left, "Extra", self.show_extra, self.transparency_extra, "#0090ff", "Blue")

        ttk.Separator(left).pack(fill=tk.X, pady=1)

        # ---------- Editing Section ----------
        ttk.Label(left, text="Editing", font=bold_font).pack(anchor=tk.W)

        tk.Button(left, text="Delete All GT", bg="#cc0a0a", fg="black",command=self.delete_all_gt_for_current).pack(fill=tk.X, pady=2)
        tk.Button(left, text="Delete Image & Label", bg="#cc0a0a", fg="black",command=self.delete_current_image_and_label).pack(fill=tk.X, pady=2)
        tk.Button(left, text="Validate All Extra → GT", bg="#28a745", fg="black",command=self.validate_all_extra_to_gt).pack(fill=tk.X, pady=2)
        tk.Button(left, text="Save GT (overwrite .txt)", bg="#28a745", fg="black",command=self.save_current_annotations).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=2)


                # ---------- Options & Zoom ----------
        ttk.Label(left, text="Navigation", font=bold_font).pack(anchor=tk.W)
        
        # ---- Pan controls (arrows) ----
        pan_frame = ttk.Frame(left)
        pan_frame.pack(pady=(6, 2), anchor="center")  # Center the whole frame horizontally

        # ttk.Label(pan_frame, text="Pan").grid(row=0, column=0, columnspan=3)

        # Arrow buttons

        arrow_font = font.Font(family="TkDefaultFont", size=13, weight="bold")  # adjust size as needed

        # Arrow buttons with bigger font
        tk.Button(pan_frame, text="↑", width=3, bg="#28a745", fg="black", font=arrow_font,command=lambda: self.pan(0, 1)).grid(row=1, column=1)
        tk.Button(pan_frame, text="←", width=3, bg="#28a745", fg="black", font=arrow_font,command=lambda: self.pan(1, 0)).grid(row=2, column=0)
        tk.Button(pan_frame, text="→", width=3, bg="#28a745", fg="black", font=arrow_font,command=lambda: self.pan(-1, 0)).grid(row=2, column=2)
        tk.Button(pan_frame, text="↓", width=3, bg="#28a745", fg="black", font=arrow_font,command=lambda: self.pan(0, -1)).grid(row=3, column=1)
        # Center the grid content
        for i in range(3):
            pan_frame.columnconfigure(i, weight=1)
        for i in range(4):
            pan_frame.rowconfigure(i, weight=1)

        ttk.Button(left, text="Fit to Window", command=self.fit_to_window).pack(fill=tk.X, pady=5)
        # ---- MAIN + RIGHT TOOL PANEL AREA ----
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Navigation bar (top)
        nav_frame = tk.Frame(right, bg="#222")
        nav_frame.pack(fill=tk.X, pady=(6, 2))
        nav_frame.columnconfigure(1, weight=1)
        tk.Button(nav_frame, text="◀ Prev", command=self.prev_image, bg="#444", fg="white",
                  font=("Segoe UI", 10, "bold")).grid(row=0, column=0, padx=2, pady=2)
        self.lbl_image_name = tk.Label(nav_frame, text="No image", fg="white", bg="#222", font=("Segoe UI", 10, "bold"))
        self.lbl_image_name.grid(row=0, column=1, sticky="ew")
        tk.Button(nav_frame, text="Next ▶", command=self.next_image, bg="#444", fg="white",
                  font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=2, pady=2)

        # Image slider
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = tk.Scale(nav_frame, from_=0, to=0, orient="horizontal", variable=self.slider_var,
                               command=self._on_slider_move, showvalue=False, length=360)
        self.slider.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 1))

        # Center area: image canvas + right tool panel
        image_area = ttk.Frame(right)
        image_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # # # Canvas in the center
        self.canvas = tk.Canvas(image_area, bg="#222222", cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        # --- TOOL SECTION ON THE RIGHT ---
        tool_frame = ttk.Frame(image_area, width=260, relief=tk.GROOVE, borderwidth=1)
        tool_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Main Tool Category selector ---
        ttk.Label(tool_frame, text="Tool Category", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(4, 2))
        self.tool_category = tk.StringVar(value="YOLO Prediction")

        category_frame = ttk.Frame(tool_frame)
        category_frame.pack(fill=tk.X, pady=(2, 1))
        ttk.Radiobutton(category_frame, text="YOLO Prediction", variable=self.tool_category,
                        value="YOLO Prediction", command=self._rebuild_tool_panel).pack(anchor=tk.W)
        ttk.Radiobutton(category_frame, text="Machine Vision", variable=self.tool_category,
                        value="Machine Vision", command=self._rebuild_tool_panel).pack(anchor=tk.W)

        ttk.Separator(tool_frame).pack(fill=tk.X, pady=4)

        # --- Dynamic area for YOLO/Machine Vision panels ---
        self.tool_panel = ttk.Frame(tool_frame)
        self.tool_panel.pack(fill=tk.BOTH, expand=True, pady=(6, 6))

        # Build initial tool panel
        self._rebuild_tool_panel()


        # Canvas bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-2>", self.on_middle_mouse_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_mouse_move)
        self.canvas.bind("<Configure>", lambda e: self.fit_to_window())
        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)      # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel_zoom)        # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel_zoom)        # Linux scroll down

        # Middle or left mouse drag for panning
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)         # Middle mouse
        self.canvas.bind("<B2-Motion>", self._on_pan_move)

        # Initial visuals
        self.update_mode_buttons_look()


    # ---------- Adaptive tool panel ----------
    def _on_tool_select(self, event=None):
        sel = self.tool_listbox.curselection()
        if sel:
            idx = sel[0]
            tool = self.tools[idx]
            self.selected_tool.set(tool)
            self._rebuild_tool_panel()

   
       # ---------- Dynamic tool panel ----------
    
    def _rebuild_tool_panel(self):
        # Clear dynamic content
        for child in self.tool_panel.winfo_children():
            child.destroy()

        category = self.tool_category.get()

        if category == "YOLO Prediction":
            ttk.Label(self.tool_panel, text="YOLO Prediction", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
            ttk.Button(self.tool_panel, text="Load YOLO Model", command=lambda: load_yolo_model(self)).pack(fill=tk.X, pady=2)
            ttk.Button(self.tool_panel, text="Predict Current Image", command=lambda: predict_current_image(self)).pack(fill=tk.X, pady=2)
            ttk.Button(self.tool_panel, text="Validate Selected Prediction", command=self.validate_selected_prediction).pack(fill=tk.X, pady=2)
            ttk.Button(self.tool_panel, text="Validate All Predictions", command=self.validate_all_predictions).pack(fill=tk.X, pady=2)

        elif category == "Machine Vision":
            ttk.Label(self.tool_panel, text="Machine Vision", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
            ttk.Label(self.tool_panel, text="Select Method").pack(anchor=tk.W)
            self.mv_method = tk.StringVar(value="Excessive Green")

            # Methods
            mv_frame = ttk.Frame(self.tool_panel)
            mv_frame.pack(fill=tk.X, pady=2)
            ttk.Radiobutton(mv_frame, text="Excessive Green", variable=self.mv_method,
                            value="Excessive Green", command=self._build_machine_vision_panel).pack(anchor=tk.W)
            ttk.Radiobutton(mv_frame, text="Circle Detection (Hough)", variable=self.mv_method,
                            value="Hough Circles", command=self._build_machine_vision_panel).pack(anchor=tk.W)
            ttk.Radiobutton(mv_frame, text="Square Detection", variable=self.mv_method,
                            value="Square Detection", command=self._build_machine_vision_panel).pack(anchor=tk.W)

            ttk.Separator(self.tool_panel).pack(fill=tk.X, pady=4)

            # Placeholder for per-method controls
            self.mv_panel = ttk.Frame(self.tool_panel)
            self.mv_panel.pack(fill=tk.BOTH, expand=True)
            self._build_machine_vision_panel()

    # ---------- Folder & files ----------
    def select_parent_folder(self):
        folder = filedialog.askdirectory(title="Select Parent Folder (must contain 'images' and 'labels' subfolders)")
        if not folder:
            return
        self.image_folder = os.path.join(folder, "images")
        self.label_folder = os.path.join(folder, "labels")
        self.prediction_folder = os.path.join(folder, "predictions")
        os.makedirs(self.prediction_folder, exist_ok=True)
        if not os.path.exists(self.image_folder) or not os.path.exists(self.label_folder):
            messagebox.showerror("Error", "Parent folder must contain 'images' and 'labels' subfolders.")
            return
        self.load_image_list()
        self.current_index = 0 if self.image_files else -1
        self.load_annotations_for_all_images()
        self.show_image_at_index()

    def load_image_list(self):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        files = []
        for e in exts:
            files.extend(sorted(glob.glob(os.path.join(self.image_folder, e))))
        self.image_files = files
        # reset dicts
        self.bboxes_gt = {}
        self.bboxes_pred = {}
        self.bboxes_extra = {}

    def load_annotations_for_all_images(self):
        if not self.label_folder:
            return
        for img_path in self.image_files:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.label_folder, fname + ".txt")
            bboxes = []
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r") as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                    with Image.open(img_path) as im:
                        w, h = im.size
                    for L in lines:
                        parts = L.split()
                        if len(parts) >= 5:
                            cls = int(float(parts[0]))
                            nx = float(parts[1])
                            ny = float(parts[2])
                            nw = float(parts[3])
                            nh = float(parts[4])
                            x1, y1, x2, y2 = yolo_to_pixels(nx, ny, nw, nh, w, h)
                            bboxes.append(BBox(cls, x1, y1, x2, y2))
                except Exception as e:
                    print("Error reading", txt_path, e)
            self.bboxes_gt[img_path] = bboxes
    
    def _build_machine_vision_panel(self):
        for child in self.mv_panel.winfo_children():
            child.destroy()

        method = self.mv_method.get()

        # helper: create a labeled scale with a live value label next to it
        def make_param_slider(label, var, from_, to_, length=160, fmt=None):
            ttk.Label(self.mv_panel, text=label).pack(anchor=tk.W)
            row = ttk.Frame(self.mv_panel)
            row.pack(anchor=tk.W, pady=(0, 4))
            ttk.Scale(row, from_=from_, to=to_, orient='horizontal', variable=var, length=length,
                      command=lambda e: None).pack(side=tk.LEFT)
            # value label
            if fmt is None:
                if isinstance(var, tk.DoubleVar):
                    fmt = lambda v: f"{float(v):.3f}"
                else:
                    fmt = lambda v: f"{int(float(v))}"
            val_lab = tk.Label(row, text=fmt(var.get()))
            val_lab.pack(side=tk.LEFT, padx=(6, 0))
            # update label when variable changes
            def _upd(*args):
                try:
                    val_lab.config(text=fmt(var.get()))
                except Exception:
                    pass
            try:
                var.trace_add('write', _upd)
            except Exception:
                try:
                    var.trace('w', _upd)
                except Exception:
                    pass

        if method == "Excessive Green":
            ttk.Label(self.mv_panel, text="Excessive Green (ExG)", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(self.mv_panel, text="Step 1: Compute ExG index").pack(anchor=tk.W, pady=(4, 0))
            ttk.Button(self.mv_panel, text="Run ExG", command=lambda: excessive_green_apply(self)).pack(fill=tk.X, pady=2)

            ttk.Label(self.mv_panel, text="Step 2: Threshold & Mask").pack(anchor=tk.W, pady=(6, 0))
            self.eg_thresh_var = tk.IntVar(value=50)
            ttk.Scale(self.mv_panel, from_=0, to=255, orient='horizontal',
                      variable=self.eg_thresh_var, length=160,
                      command=lambda e: self._render_image_on_canvas()).pack(anchor=tk.W)
            ttk.Button(self.mv_panel, text="Apply Threshold", command=lambda:excessive_green_apply(self)).pack(fill=tk.X, pady=4)

        elif method == "Hough Circles":
            ttk.Label(self.mv_panel, text="Hough Circle Detection", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(self.mv_panel, text="Detect circular shapes using OpenCV").pack(anchor=tk.W, pady=(4, 2))

            # preview toggle
            ttk.Checkbutton(self.mv_panel, text="Show MV preview (circles/overlays)", variable=self.show_mv_preview,
                            command=lambda: self._render_image_on_canvas()).pack(anchor=tk.W, pady=(2,4))

            # Parameters
            self.hough_param1 = tk.IntVar(value=100)   # Canny edge threshold
            self.hough_param2 = tk.IntVar(value=30)    # Accumulator threshold
            self.hough_min_r = tk.IntVar(value=10)     # Minimum radius
            self.hough_max_r = tk.IntVar(value=80)     # Maximum radius
            # distance between circle centers
            self.hough_minDist = tk.IntVar(value=20)

            make_param_slider("Canny Threshold (param1)", self.hough_param1, 10, 300)
            make_param_slider("Accumulator Threshold (param2)", self.hough_param2, 5, 100)
            make_param_slider("Min Radius", self.hough_min_r, 0, 100)
            make_param_slider("Max Radius", self.hough_max_r, 0, 200)
            make_param_slider("Min Distance between circles", self.hough_minDist, 1, 200)

            # auto-add checkbox (right side)
            self.mv_auto_add_circles = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.mv_panel, text="Auto-add to extras", variable=self.mv_auto_add_circles).pack(anchor=tk.E)

            # call MV helper with self on click (uses wrapper to optionally auto-add)
            ttk.Button(self.mv_panel, text="Run Circle Detection", command=lambda: self._mv_run_and_maybe_add('circles')).pack(fill=tk.X, pady=(6,2))
            # convert detected circles into bboxes (manual)
            ttk.Button(self.mv_panel, text="Convert detected circles → BBoxes", command=lambda: self._mv_convert_detected('detected_circles')).pack(fill=tk.X, pady=(0,6))

        elif method == "Square Detection":
            ttk.Label(self.mv_panel, text="Square Detection", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(self.mv_panel, text="Contour-based square detection using OpenCV").pack(anchor=tk.W, pady=(4, 2))
            # rectangle detection parameters
            self.rect_canny_low = tk.IntVar(value=50)
            self.rect_canny_high = tk.IntVar(value=150)
            self.rect_min_area = tk.IntVar(value=100)
            self.rect_approx_eps = tk.DoubleVar(value=0.02)
            make_param_slider("Canny low", self.rect_canny_low, 0, 200)
            make_param_slider("Canny high", self.rect_canny_high, 0, 300)
            make_param_slider("Min area", self.rect_min_area, 10, 10000)
            make_param_slider("Approx epsilon (fraction of perimeter)", self.rect_approx_eps, 0.005, 0.1, fmt=lambda v: f"{float(v):.4f}")
            # auto-add checkbox for shapes
            self.mv_auto_add_shapes = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.mv_panel, text="Auto-add to extras", variable=self.mv_auto_add_shapes).pack(anchor=tk.E)

            # call rectangle detection helper (wrapper to possibly auto-add)
            ttk.Button(self.mv_panel, text="Run Square Detection", command=lambda: self._mv_run_and_maybe_add('rects')).pack(fill=tk.X, pady=4)
            # triangle detection
            ttk.Separator(self.mv_panel).pack(fill=tk.X, pady=(6,4))
            ttk.Button(self.mv_panel, text="Run Triangle Detection", command=lambda: self._mv_run_and_maybe_add('triangles')).pack(fill=tk.X, pady=2)
            # polygon detection (N vertices)
            self.poly_n = tk.IntVar(value=5)
            make_param_slider("Polygon vertices (N)", self.poly_n, 3, 12)
            ttk.Button(self.mv_panel, text="Run Polygon Detection", command=lambda: self._mv_run_and_maybe_add('polygons')).pack(fill=tk.X, pady=4)

        # end of _build_machine_vision_panel

    def _mv_convert_detected(self, attr_name: str):
        """Generic converter: look up detected items in `self.<attr_name>` and append BBox extras for current image.

        Supported attr_name values: 'detected_circles', 'detected_rects', 'detected_triangles', 'detected_polygons'
        """
        img_path = self.image_files[self.current_index] if getattr(self, 'image_files', None) else None
        if not img_path:
            messagebox.showwarning("No image", "No image loaded to add bboxes to.")
            return
        items = getattr(self, attr_name, None)
        if not items:
            messagebox.showinfo("Nothing", "No detected items to convert.")
            return
        try:
            from core.models import BBox
        except Exception:
            BBox = None
        added = 0
        last_added = None
        if attr_name == 'detected_circles':
            for (x, y, r) in items:
                x1, y1, x2, y2 = x - r, y - r, x + r, y + r
                if BBox is not None:
                    # add as a prediction so it can be validated/deleted using the UI modes
                    self.bboxes_pred.setdefault(img_path, []).append(BBox(0, x1, y1, x2, y2))
                    last_added = self.bboxes_pred[img_path][-1]
                added += 1
        else:
            # items are stored as list of (x1,y1,x2,y2)
            for (x1, y1, x2, y2) in items:
                if BBox is not None:
                    # add as prediction by default so user can validate/delete
                    self.bboxes_pred.setdefault(img_path, []).append(BBox(0, x1, y1, x2, y2))
                    last_added = self.bboxes_pred[img_path][-1]
                added += 1
        # select the last added bbox so the user can immediately delete or validate it
        if last_added is not None:
            self.selected_bbox = last_added
        self._render_image_on_canvas()
        # refresh left panel counts
        try:
            img_path = self.image_files[self.current_index]
            gt_count = len(self.bboxes_gt.get(img_path, []))
            pred_count = len(self.bboxes_pred.get(img_path, []))
            extra_count = len(self.bboxes_extra.get(img_path, []))
            self.lbl_counts.config(text=f"GT: {gt_count}   Pred: {pred_count}   Extra: {extra_count}")
            dc = getattr(self, 'detected_circles', []) or []
            dr = getattr(self, 'detected_rects', []) or []
            dt = getattr(self, 'detected_triangles', []) or []
            dp = getattr(self, 'detected_polygons', []) or []
            self.lbl_detected.config(text=f"Detected - Circles: {len(dc)}  Rects: {len(dr)}  Tris: {len(dt)}  Polys: {len(dp)}")
        except Exception:
            pass
        messagebox.showinfo("Converted", f"Converted {added} item(s) to bbox extras.")

    def _mv_run_and_maybe_add(self, kind: str):
        """Run the appropriate detector and optionally auto-convert results to extras.

        kind: 'circles', 'rects', 'triangles', 'polygons'
        """
        if kind == 'circles':
            circle_detect(self)
            if getattr(self, 'mv_auto_add_circles', tk.BooleanVar(value=False)).get():
                self._mv_convert_detected('detected_circles')
        elif kind == 'rects':
            rectangle_detect(self)
            if getattr(self, 'mv_auto_add_shapes', tk.BooleanVar(value=False)).get():
                self._mv_convert_detected('detected_rects')
        elif kind == 'triangles':
            triangle_detect(self)
            if getattr(self, 'mv_auto_add_shapes', tk.BooleanVar(value=False)).get():
                self._mv_convert_detected('detected_triangles')
        elif kind == 'polygons':
            polygon_detect(self)
            if getattr(self, 'mv_auto_add_shapes', tk.BooleanVar(value=False)).get():
                self._mv_convert_detected('detected_polygons')

    # ---------- Image loading & showing ----------
    def load_current_image(self):
        if not self.image_files or self.current_index < 0:
            self.current_image = None
            return
        img_path = self.image_files[self.current_index]
        try:
            img = Image.open(img_path).convert("RGB")
            self.current_image = img
            self.img_w, self.img_h = img.size
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{img_path}\n{e}")
            self.current_image = None

    def show_image_at_index(self):
        """Show image at self.current_index and update UI elements."""
        if not self.image_files or self.current_index < 0:
            self.lbl_image_name.config(text="No image")
            self.canvas.delete("all")
            self.slider.config(to=0)
            # reset data info labels if they exist
            if hasattr(self, "lbl_shape_info"):
                self.lbl_shape_info.config(text="Shape: -")
            if hasattr(self, "lbl_counts_info"):
                self.lbl_counts_info.config(text="GT: 0   Pred: 0   Extra: 0")
            return

        # clamp index
        self.current_index = max(0, min(self.current_index, len(self.image_files) - 1))
        self.load_current_image()

        # clear previous machine-vision preview/detections
        self.mv_preview_image = None
        self.detected_circles = []
        self.mv_tk_image = None

        if self.image_files:
            img_name = os.path.basename(self.image_files[self.current_index])
            self.lbl_image_name.config(text=f"{self.current_index + 1}/{len(self.image_files)}  {img_name}")

            # Update left data panel (two rows)
            try:
                img_w, img_h = self.img_w, self.img_h
                gt_count = len(self.bboxes_gt.get(self.image_files[self.current_index], []))
                pred_count = len(self.bboxes_pred.get(self.image_files[self.current_index], []))
                extra_count = len(self.bboxes_extra.get(self.image_files[self.current_index], []))

                if hasattr(self, "lbl_shape_info"):
                    self.lbl_shape_info.config(text=f"Shape: {img_w} x {img_h}")

                if hasattr(self, "lbl_counts_info"):
                    self.lbl_counts_info.config(
                        text=f"GT: {gt_count}   Pred: {pred_count}   Extra: {extra_count}"
                    )

                # Detected shapes (optional hidden label)
                dc = getattr(self, 'detected_circles', []) or []
                dr = getattr(self, 'detected_rects', []) or []
                dt = getattr(self, 'detected_triangles', []) or []
                dp = getattr(self, 'detected_polygons', []) or []
                if hasattr(self, "lbl_detected"):
                    self.lbl_detected.config(
                        text=f"Detected - Circles: {len(dc)}  Rects: {len(dr)}  Tris: {len(dt)}  Polys: {len(dp)}"
                    )
            except Exception:
                pass

        # reset offset preservation so image gets centered
        self._preserve_offset = False

        # update slider range and render
        self.update_slider_range()
        self._render_image_on_canvas()



    def update_slider_range(self):
        n = max(1, len(self.image_files))
        self.slider.config(to=max(0, n - 1))
        if self.image_files:
            pos = int(self.current_index)
            self.slider_var.set(pos)

    # ---------- Drawing / rendering ----------
    def _render_image_on_canvas(self, *args):
        """Main render: draw image, overlays (PIL), and interactive outlines (canvas)."""
        self.canvas.delete("all")
        self.handle_id_to_info.clear()

        if not self.current_image:
            return

        c_w = self.canvas.winfo_width() or 800
        c_h = self.canvas.winfo_height() or 600
        if c_w < 10 or c_h < 10:
            return

        # base scale
        if self.check_keep_ratio.get():
            base_scale = min(c_w / max(1, self.img_w), c_h / max(1, self.img_h))
        else:
            base_scale = c_w / max(1, self.img_w)
        self.display_scale = base_scale * self.zoom_factor

        disp_w = max(1, int(self.img_w * self.display_scale))
        disp_h = max(1, int(self.img_h * self.display_scale))
        # center by default, but preserve offsets if a zoom operation explicitly set them
        if not getattr(self, '_preserve_offset', False):
            # use float centering to avoid integer truncation shifts when window > image
            self.offset_x = (c_w - disp_w) / 2.0
            self.offset_y = (c_h - disp_h) / 2.0

        # resized image as RGBA
        img_resized = self.current_image.resize((disp_w, disp_h), Image.LANCZOS).convert("RGBA")
        overlay = Image.new("RGBA", (disp_w, disp_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        img_path = self.image_files[self.current_index]
        gt_boxes = self.bboxes_gt.get(img_path, [])
        pred_boxes = self.bboxes_pred.get(img_path, [])
        extra_boxes = []

        # compute extra = preds unmatched (IoU <= 0.5)
        for p in pred_boxes:
            matched = False
            for g in gt_boxes:
                if calculate_iou(p.as_tuple(), g.as_tuple()) > 0.5:
                    matched = True
                    break
            if not matched:
                extra_boxes.append(p)

        # also include any manually stored extras
        manual_extras = self.bboxes_extra.get(img_path, [])
        extra_boxes.extend(manual_extras)

        # Helper to draw PIL rectangles (fills + outlines that exactly match transparency)
        # We draw fills and outlines on the overlay image (PIL) so both support alpha.
        def draw_pil_boxes(box_list, fill_rgb, trans_var, outline_width=2, glow=True):
            alpha = max(0.0, min(1.0, trans_var.get()))
            if alpha <= 0.0:
                return
            fill_a = int(255 * alpha)
            outline_a = int(255 * alpha)
            for bb in box_list:
                x1 = max(0, int(bb.x1 * self.display_scale))
                y1 = max(0, int(bb.y1 * self.display_scale))
                x2 = min(disp_w, int(bb.x2 * self.display_scale))
                y2 = min(disp_h, int(bb.y2 * self.display_scale))
                # draw main filled rect with outline (Pillow supports RGBA outlines)
                try:
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], fill_a),
                        outline=(fill_rgb[0], fill_rgb[1], fill_rgb[2], outline_a),
                        width=outline_width
                    )
                except TypeError:
                    # older Pillow versions may not support `width` for outline; fallback:
                    draw.rectangle([x1, y1, x2, y2], fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], fill_a))
                    # draw a single-pixel outline manually
                    draw.rectangle([x1, y1, x2, y1], fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], outline_a))
                    draw.rectangle([x1, y2, x2, y2], fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], outline_a))
                    draw.rectangle([x1, y1, x1, y2], fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], outline_a))
                    draw.rectangle([x2, y1, x2, y2], fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], outline_a))

                # optional soft glow (draw larger translucent outlines to simulate glow)
                if glow:
                    # glow uses lower alpha and increasing size
                    for expand, g_mul in ((1, 0.6), (3, 0.28)):
                        g_alpha = int(outline_a * g_mul)
                        if g_alpha <= 0:
                            continue
                        draw.rectangle(
                            [x1 - expand, y1 - expand, x2 + expand, y2 + expand],
                            outline=(fill_rgb[0], fill_rgb[1], fill_rgb[2], g_alpha)
                        )

        # Draw GT, Predictions, Extra on overlay (so they alpha-blend correctly)
        if self.show_gt.get():
            draw_pil_boxes(gt_boxes, (0, 200, 0), self.transparency_gt)
        if self.show_pred.get():
            draw_pil_boxes(pred_boxes, (255, 80, 80), self.transparency_pred)
        if self.show_extra.get():
            draw_pil_boxes(extra_boxes, (0, 140, 255), self.transparency_extra)

        # If Excessive Green preview active and tool selected, draw mask overlay
        if self.selected_tool.get() == "Excessive Green" and getattr(self, "eg_preview_var", None) and self.eg_preview_var.get():
            # compute mask quickly
            thresh = getattr(self, "eg_thresh_var", tk.IntVar(value=50)).get()
            mask = self._compute_excessive_green_mask(self.current_image, thresh)
            if mask is not None:
                # mask is same size as original image -> resize to disp and put translucent magenta overlay
                mask_resized = mask.resize((disp_w, disp_h), Image.NEAREST).convert("L")
                mask_overlay = Image.new("RGBA", (disp_w, disp_h), (0, 0, 0, 0))
                mo_draw = ImageDraw.Draw(mask_overlay, "RGBA")
                pix = mask_resized.load()
                for yy in range(disp_h):
                    for xx in range(disp_w):
                        if pix[xx, yy] > 0:
                            mo_draw.point((xx, yy), fill=(200, 0, 200, 80))
                overlay = Image.alpha_composite(overlay, mask_overlay)

        # Composite overlay onto image
        combined = Image.alpha_composite(img_resized, overlay)

        # display on canvas (tag as base image so previews can be positioned)
        self.tk_image = ImageTk.PhotoImage(combined)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image, tags=("base_img",))

        # Draw interactive outlines in canvas — these are for interactivity.
        # IMPORTANT: Tkinter Canvas does NOT support alpha in item colors. If we draw opaque canvas rectangles,
        # they will sit on top of the PIL overlay and hide the faded outlines. To ensure the visible border always
        # matches the faded fill, we:
        #  - skip drawing opaque canvas outlines when the transparency is not ~1.0 (so the PIL overlay outlines show)
        #  - draw canvas outlines only when transparency is nearly fully opaque (alpha >= 0.99)
        # This preserves both correct faded visuals and interactivity when fully visible.
        def draw_outline_list(box_list, base_color, transparency_var):
            alpha_val = max(0.0, min(1.0, transparency_var.get()))
            # if nearly opaque, draw canvas outlines for crisp interactive feedback
            if alpha_val >= 0.99:
                # map alpha_val [0..1] to brightness factor (keeps similar to before)
                brightness = 0.35 + 0.65 * alpha_val
                r, g, b = base_color
                adj = (int(r * brightness), int(g * brightness), int(b * brightness))
                adj_color = f"#{adj[0]:02x}{adj[1]:02x}{adj[2]:02x}"
                for bb in box_list:
                    x1 = bb.x1 * self.display_scale + self.offset_x
                    y1 = bb.y1 * self.display_scale + self.offset_y
                    x2 = bb.x2 * self.display_scale + self.offset_x
                    y2 = bb.y2 * self.display_scale + self.offset_y
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline=adj_color, width=2, tags=("bbox", f"bbox_{id(bb)}"))
                    self.canvas.create_text(x1 + 4, y1 + 4, anchor=tk.NW, text=str(bb.cls),
                                            fill=adj_color, font=("TkDefaultFont", 10, "bold"),
                                            tags=("bbox", f"bbox_{id(bb)}"))
            else:
                # transparency < 0.99 -> rely on the PIL overlay for visible border (already drawn with alpha)
                # but still create invisible/interactable bbox items if you need events: create rectangle with empty outline and proper tags
                for bb in box_list:
                    x1 = bb.x1 * self.display_scale + self.offset_x
                    y1 = bb.y1 * self.display_scale + self.offset_y
                    x2 = bb.x2 * self.display_scale + self.offset_x
                    y2 = bb.y2 * self.display_scale + self.offset_y
                    # invisible rectangle for clicks/hover (outline="" and stipple/width=0)
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="", width=0, tags=("bbox", f"bbox_{id(bb)}"))

        if self.show_gt.get():
            draw_outline_list(gt_boxes, (0, 200, 0), self.transparency_gt)
        if self.show_pred.get():
            draw_outline_list(pred_boxes, (255, 80, 80), self.transparency_pred)
        if self.show_extra.get():
            draw_outline_list(extra_boxes, (0, 140, 255), self.transparency_extra)

        # draw handles for selected bbox
        if self.selected_bbox:
            if (self.selected_bbox in gt_boxes) or (self.selected_bbox in pred_boxes) or (self.selected_bbox in extra_boxes):
                # Handles are interactive and should remain visible regardless of mask alpha.
                # Draw them on the canvas (keep as-is).
                self._draw_handles(self.selected_bbox)

        # Show Hough Circle result if available
        if getattr(self, 'show_mv_preview', tk.BooleanVar(value=True)).get() and hasattr(self, "mv_preview_image") and self.mv_preview_image is not None:
            try:
                self._display_image_on_canvas(self.mv_preview_image)
            except Exception:
                pass
            return


    def _draw_handles(self, bbox: BBox):
        # draw small squares at the corners on top of canvas
        x1 = bbox.x1 * self.display_scale + self.offset_x
        y1 = bbox.y1 * self.display_scale + self.offset_y
        x2 = bbox.x2 * self.display_scale + self.offset_x
        y2 = bbox.y2 * self.display_scale + self.offset_y
        s = self.HANDLE_SIZE
        coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i, (cx, cy) in enumerate(coords):
            hid = self.canvas.create_rectangle(cx - s, cy - s, cx + s, cy + s, fill="#FFFFFF", outline="#000000", tags=("handle", f"bbox_{id(bbox)}", str(i)))
            self.handle_id_to_info[hid] = (bbox, i)

    def _display_image_on_canvas(self, img):
        """Display a preview image (numpy array or PIL Image) on the canvas.

        Accepts an OpenCV BGR numpy array or a PIL Image. Scales and positions the
        preview to the same display size as the main image and draws it on top.
        """
        try:
            # convert numpy (BGR or RGB) to PIL
            from PIL import Image
            import numpy as np
        except Exception:
            Image = None
            np = None

        pil_img = None
        # if it's a numpy array, convert
        if np is not None and isinstance(img, np.ndarray):
            # assume BGR -> convert to RGB
            try:
                import cv2
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                # fallback: assume already RGB
                rgb = img
            pil_img = Image.fromarray(rgb)
        elif Image is not None and isinstance(img, Image.Image):
            pil_img = img

        if pil_img is None:
            return

            # resize preview to current display size
        try:
            disp_w = max(1, int(self.img_w * self.display_scale))
            disp_h = max(1, int(self.img_h * self.display_scale))
            pil_resized = pil_img.resize((disp_w, disp_h), Image.LANCZOS).convert("RGB")
            # create ImageTk and draw onto canvas at same offsets
            self.mv_tk_image = ImageTk.PhotoImage(pil_resized)
            # place preview over existing image; use a tag so it can be removed next render
            self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.mv_tk_image, tags=("mv_preview",))
            # place the preview between the base image and any bbox overlays.
            # If there are bbox items, lower the preview beneath them so bboxes stay on top.
            # Otherwise, ensure the preview is above the base image so it's visible.
            try:
                if self.canvas.find_withtag("bbox"):
                    try:
                        self.canvas.tag_lower("mv_preview", "bbox")
                    except Exception:
                        pass
                else:
                    try:
                        self.canvas.tag_raise("mv_preview", "base_img")
                    except Exception:
                        try:
                            self.canvas.tag_raise("mv_preview")
                        except Exception:
                            pass
            except Exception:
                # fallback: try to raise the preview so it's visible
                try:
                    self.canvas.tag_raise("mv_preview")
                except Exception:
                    pass
        except Exception:
            return

    def _mv_convert_circles_to_bboxes(self):
        """Convert any detected circles (self.detected_circles) into BBox extras for current image."""
        img_path = self.image_files[self.current_index] if getattr(self, 'image_files', None) else None
        if not img_path:
            messagebox.showwarning("No image", "No image loaded to add bboxes to.")
            return
        circles = getattr(self, 'detected_circles', None)
        if not circles:
            messagebox.showinfo("No circles", "No detected circles to convert.")
            return
        try:
            from core.models import BBox
        except Exception:
            BBox = None
        added = 0
        for (x, y, r) in circles:
            x1, y1, x2, y2 = x - r, y - r, x + r, y + r
            if BBox is not None:
                self.bboxes_extra.setdefault(img_path, []).append(BBox(0, x1, y1, x2, y2))
            added += 1
        self._render_image_on_canvas()
        messagebox.showinfo("Converted", f"Converted {added} circle(s) to bbox extras.")

    # ---------- Mouse interactions ----------
    def canvas_to_image_coords(self, cx, cy):
        ix = (cx - self.offset_x) / max(1e-6, self.display_scale)
        iy = (cy - self.offset_y) / max(1e-6, self.display_scale)
        ix = max(0, min(self.img_w, ix))
        iy = max(0, min(self.img_h, iy))
        return ix, iy
    
    def _on_mousewheel_zoom(self, event):
        """Zoom in/out at the cursor position using the mouse wheel."""
        # Normalize scroll direction across OSs
        if event.num == 5 or event.delta < 0:
            factor = 0.7   # zoom out
        else:
            factor = 1.3   # zoom in

        # Get cursor coordinates on the canvas
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # Call your existing zoom_at_cursor method
        self.zoom_at_cursor(cx, cy, factor)
    
    def _on_pan_start(self, event):
        """Record the start position for panning."""
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _on_pan_move(self, event):
        """Move the image as the mouse drags."""
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        self._pan_start_x = event.x
        self._pan_start_y = event.y

        # Call your existing pan method if it expects deltas in canvas pixels
        self.pan(dx, dy)

    def on_mouse_down(self, event):
        x, y = event.x, event.y
        mode = self.current_mode.get()

        # check for handle click
        items = self.canvas.find_overlapping(x, y, x, y)
        for it in items:
            if it in self.handle_id_to_info:
                bbox, handle_idx = self.handle_id_to_info[it]
                self.selected_bbox = bbox
                self.drag_data.update({"mode": "resize", "start": (x, y), "bbox": bbox, "handle": handle_idx})
                self._render_image_on_canvas()
                return

        # find bbox under cursor (top-most)
        clicked_bbox = None
        img_path = self.image_files[self.current_index] if self.image_files else None
        candidates = []
        if img_path:
            candidates.extend(self.bboxes_gt.get(img_path, []))
            candidates.extend(self.bboxes_pred.get(img_path, []))
            candidates.extend(self.bboxes_extra.get(img_path, []))
        for bb in reversed(candidates):
            x1 = bb.x1 * self.display_scale + self.offset_x
            y1 = bb.y1 * self.display_scale + self.offset_y
            x2 = bb.x2 * self.display_scale + self.offset_x
            y2 = bb.y2 * self.display_scale + self.offset_y
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_bbox = bb
                break

        if mode == "draw":
            self._drawing_mode = True
            self._new_rect_start = (x, y)
            self._new_rect_id = self.canvas.create_rectangle(x, y, x, y, outline="#00CCCC", width=2, dash=(3,3))
            self.drag_data.update({"mode": "draw", "start": (x, y)})
            return

        if mode == "delete":
            if clicked_bbox:
                if messagebox.askyesno("Delete bbox", f"Delete selected bbox (class {clicked_bbox.cls})?"):
                    if clicked_bbox in self.bboxes_gt.get(img_path, []):
                        self.bboxes_gt[img_path].remove(clicked_bbox)
                    if clicked_bbox in self.bboxes_pred.get(img_path, []):
                        self.bboxes_pred[img_path].remove(clicked_bbox)
                    if clicked_bbox in self.bboxes_extra.get(img_path, []):
                        self.bboxes_extra[img_path].remove(clicked_bbox)
                    if self.selected_bbox is clicked_bbox:
                        self.selected_bbox = None
                    self._render_image_on_canvas()
            return

        if mode == "validate":
            if clicked_bbox:
                img_path = self.image_files[self.current_index]
                moved = False
                # If it's a prediction, move from preds -> gt
                if clicked_bbox in self.bboxes_pred.get(img_path, []):
                    try:
                        self.bboxes_pred[img_path].remove(clicked_bbox)
                    except Exception:
                        pass
                    self.bboxes_gt.setdefault(img_path, []).append(clicked_bbox)
                    moved = True
                # If it's a manual extra, allow validating it as well
                elif clicked_bbox in self.bboxes_extra.get(img_path, []):
                    try:
                        self.bboxes_extra[img_path].remove(clicked_bbox)
                    except Exception:
                        pass
                    self.bboxes_gt.setdefault(img_path, []).append(clicked_bbox)
                    moved = True
                if moved:
                    self.selected_bbox = None
                    self._render_image_on_canvas()
            return

        if mode == "zoom":
            # start zoom rectangle
            self._zoom_rect_start = (x, y)
            self._zoom_rect_id = self.canvas.create_rectangle(x, y, x, y, outline="#AAAAFF", dash=(3,3))
            self.drag_data.update({"mode": "zoomrect", "start": (x, y)})
            return

        # default move/select
        if (mode == "move") or (mode == "none"):
            if clicked_bbox:
                self.selected_bbox = clicked_bbox
                self.drag_data.update({"mode": "move", "start": (x, y), "bbox": clicked_bbox, "handle": None})
                self._render_image_on_canvas()
                return

        if clicked_bbox is None:
            self.selected_bbox = None
            self._render_image_on_canvas()

    def on_mouse_move(self, event):
        x, y = event.x, event.y
        mode = self.drag_data.get("mode")
        if mode == "draw":
            sx, sy = self._new_rect_start
            if self._new_rect_id:
                self.canvas.coords(self._new_rect_id, sx, sy, x, y)
        elif mode == "move":
            bbox = self.drag_data.get("bbox")
            sx, sy = self.drag_data.get("start")
            if not bbox:
                return
            dx = (x - sx) / max(1e-9, self.display_scale)
            dy = (y - sy) / max(1e-9, self.display_scale)
            nx1 = bbox.x1 + dx
            ny1 = bbox.y1 + dy
            nx2 = bbox.x2 + dx
            ny2 = bbox.y2 + dy
            w = nx2 - nx1
            h = ny2 - ny1
            if nx1 < 0:
                nx1 = 0; nx2 = w
            if ny1 < 0:
                ny1 = 0; ny2 = h
            if nx2 > self.img_w:
                nx2 = self.img_w; nx1 = nx2 - w
            if ny2 > self.img_h:
                ny2 = self.img_h; ny1 = ny2 - h
            bbox.x1, bbox.y1, bbox.x2, bbox.y2 = nx1, ny1, nx2, ny2
            self.drag_data["start"] = (x, y)
            self._render_image_on_canvas()
        elif mode == "resize":
            bbox = self.drag_data.get("bbox")
            handle_idx = self.drag_data.get("handle")
            if bbox is None:
                return
            ix, iy = self.canvas_to_image_coords(x, y)
            if handle_idx == 0:
                bbox.x1 = min(ix, bbox.x2 - 1)
                bbox.y1 = min(iy, bbox.y2 - 1)
            elif handle_idx == 1:
                bbox.x2 = max(ix, bbox.x1 + 1)
                bbox.y1 = min(iy, bbox.y2 - 1)
            elif handle_idx == 2:
                bbox.x2 = max(ix, bbox.x1 + 1)
                bbox.y2 = max(iy, bbox.y1 + 1)
            elif handle_idx == 3:
                bbox.x1 = min(ix, bbox.x2 - 1)
                bbox.y2 = max(iy, bbox.y1 + 1)
            self._render_image_on_canvas()
        elif mode == "zoomrect":
            sx, sy = self._zoom_rect_start
            if hasattr(self, "_zoom_rect_id") and self._zoom_rect_id:
                self.canvas.coords(self._zoom_rect_id, sx, sy, x, y)

    def on_mouse_up(self, event):
        x, y = event.x, event.y
        mode = self.drag_data.get("mode")

        if mode == "draw":
            x1, y1 = self._new_rect_start
            x2, y2 = event.x, event.y
            if self._new_rect_id:
                try: self.canvas.delete(self._new_rect_id)
                except: pass
                self._new_rect_id = None
            ix1, iy1 = self.canvas_to_image_coords(x1, y1)
            ix2, iy2 = self.canvas_to_image_coords(x2, y2)
            a1, a2 = min(ix1, ix2), min(iy1, iy2)
            b1, b2 = max(ix1, ix2), max(iy1, iy2)
            if abs(b1 - a1) < 3 or abs(b2 - a2) < 3:
                self.drag_data = {"mode": None}; self._drawing_mode = False
                return
            try:
                cls_val = int(getattr(self, "cls_var", tk.StringVar(value="0")).get())
            except:
                cls_val = 0
            new_bbox = BBox(cls_val, a1, a2, b1, b2)
            self.bboxes_gt.setdefault(self.image_files[self.current_index], []).append(new_bbox)
            self.selected_bbox = new_bbox
            self._drawing_mode = False
            self.drag_data = {"mode": None}
            self._render_image_on_canvas()
            return

        if mode == "zoomrect":
            if hasattr(self, "_zoom_rect_id") and self._zoom_rect_id:
                coords = self.canvas.coords(self._zoom_rect_id)
                try: self.canvas.delete(self._zoom_rect_id)
                except: pass
                self._zoom_rect_id = None
                if coords and len(coords) == 4:
                    x1, y1, x2, y2 = coords
                    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                        self.zoom_to_selection(x1, y1, x2, y2)
            self.drag_data = {"mode": None}
            return

        # otherwise clear drag state
        self.drag_data = {"mode": None, "start": (0,0), "bbox": None, "handle": None}

    def on_right_click(self, event):
        x, y = event.x, event.y
        clicked_bbox = None
        img_path = self.image_files[self.current_index] if self.image_files else None
        candidates = []
        if img_path:
            candidates.extend(self.bboxes_gt.get(img_path, []))
            candidates.extend(self.bboxes_pred.get(img_path, []))
            candidates.extend(self.bboxes_extra.get(img_path, []))
        for bb in reversed(candidates):
            x1 = bb.x1 * self.display_scale + self.offset_x
            y1 = bb.y1 * self.display_scale + self.offset_y
            x2 = bb.x2 * self.display_scale + self.offset_x
            y2 = bb.y2 * self.display_scale + self.offset_y
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_bbox = bb
                break
        if clicked_bbox:
            self.selected_bbox = clicked_bbox
            self._render_image_on_canvas()

    def on_mouse_wheel(self, event):
        try:
            ctrl_pressed = (event.state & 0x0004) != 0
        except Exception:
            ctrl_pressed = False
        if ctrl_pressed:
            factor = 1.15 if event.delta > 0 else 1/1.15
            self.zoom_at_cursor(event.x, event.y, factor)

    def on_middle_mouse_down(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_middle_mouse_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # ---------- Zoom helpers ----------

    def fit_to_window(self):
        """Reset zoom and center the image to fit the canvas."""
        if not self.current_image:
            return

        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1

        # Compute base scale depending on aspect ratio mode
        if self.check_keep_ratio.get():
            base_scale = min(c_w / max(1, self.img_w), c_h / max(1, self.img_h))
        else:
            base_scale = c_w / max(1, self.img_w)

        # Reset scale and offsets
        self.zoom_factor = 1.0
        self.display_scale = base_scale

        # Center image in the canvas
        disp_w = self.img_w * base_scale
        disp_h = self.img_h * base_scale
        self.offset_x = (c_w - disp_w) / 2
        self.offset_y = (c_h - disp_h) / 2

        # Make sure render doesn’t override this offset
        self._preserve_offset = True

        self._render_image_on_canvas()

    def zoom_at_center(self, factor):
        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        self.zoom_at_cursor(c_w//2, c_h//2, factor)

    def zoom_at_cursor(self, cx, cy, factor):
        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        if self.check_keep_ratio.get():
            base_scale = min(c_w / max(1, self.img_w), c_h / max(1, self.img_h))
        else:
            base_scale = c_w / max(1, self.img_w)
        # Clamp cursor to visible image area
        disp_w = int(self.img_w * self.display_scale) if getattr(self, 'display_scale', None) else int(self.img_w * base_scale * getattr(self, 'zoom_factor', 1.0))
        disp_h = int(self.img_h * self.display_scale) if getattr(self, 'display_scale', None) else int(self.img_h * base_scale * getattr(self, 'zoom_factor', 1.0))
        img_left = getattr(self, 'offset_x', (c_w - disp_w) / 2.0)
        img_top = getattr(self, 'offset_y', (c_h - disp_h) / 2.0)
        img_right = img_left + disp_w
        img_bottom = img_top + disp_h
        cx = max(img_left, min(img_right, cx))
        cy = max(img_top, min(img_bottom, cy))
        # image coords under cursor before zoom
        ix, iy = self.canvas_to_image_coords(cx, cy)
        # Store debug anchor for rendering
        self._debug_zoom_anchor = (cx, cy, ix, iy)
        self.zoom_factor *= factor
        new_display_scale = base_scale * self.zoom_factor
        # compute new offsets so the image point (ix,iy) remains under the cursor
        self.offset_x = cx - ix * new_display_scale
        self.offset_y = cy - iy * new_display_scale
        # preserve offsets so _render_image_on_canvas won't recenter
        self._preserve_offset = True
        self._render_image_on_canvas()

    def zoom_to_selection(self, x1, y1, x2, y2):
        """Zoom so that region defined by two canvas points fits the canvas centered."""
        if not self.current_image:
            return
        # normalize and convert to image coords
        x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
        # Clamp selection to the currently visible image area so dragging outside doesn't produce odd zoom centers
        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        disp_w = max(1, int(self.img_w * getattr(self, 'display_scale', 1.0)))
        disp_h = max(1, int(self.img_h * getattr(self, 'display_scale', 1.0)))
        left = getattr(self, 'offset_x', (c_w - disp_w) / 2.0)
        top = getattr(self, 'offset_y', (c_h - disp_h) / 2.0)
        right = left + disp_w
        bottom = top + disp_h
        # clamp coordinates
        x1 = max(left, min(right, x1))
        x2 = max(left, min(right, x2))
        y1 = max(top, min(bottom, y1))
        y2 = max(top, min(bottom, y2))
        img_x1 = (x1 - self.offset_x) / max(1e-9, self.display_scale)
        img_y1 = (y1 - self.offset_y) / max(1e-9, self.display_scale)
        img_x2 = (x2 - self.offset_x) / max(1e-9, self.display_scale)
        img_y2 = (y2 - self.offset_y) / max(1e-9, self.display_scale)
        region_w = max(1, abs(img_x2 - img_x1))
        region_h = max(1, abs(img_y2 - img_y1))
        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        # compute new display scale so region fills the canvas (may crop one axis)
        # use max so the region is scaled up to fully occupy the canvas in at least one axis
        new_display_scale = max(c_w / region_w, c_h / region_h)
        # update zoom_factor relative to base_scale
        if self.check_keep_ratio.get():
            base_scale = min(c_w / max(1, self.img_w), c_h / max(1, self.img_h))
        else:
            base_scale = c_w / max(1, self.img_w)
        self.zoom_factor = new_display_scale / base_scale
        # offset so the selected image region's top-left maps to centered region on canvas
        # compute region pixel size under new scale
        region_pixel_w = region_w * new_display_scale
        region_pixel_h = region_h * new_display_scale
        self.offset_x = (c_w - region_pixel_w) / 2 - min(img_x1, img_x2) * new_display_scale
        self.offset_y = (c_h - region_pixel_h) / 2 - min(img_y1, img_y2) * new_display_scale
        self._render_image_on_canvas()

    def pan(self, dx_dir, dy_dir, step_px: int = None):
        """Pan the displayed image by a directional step.

        dx_dir/dy_dir should be -1, 0 or 1 indicating direction. step_px if provided is pixel step
        in canvas coords; otherwise computed as 20% of the smaller canvas dimension.
        """
        # compute step
        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        if step_px is None:
            step = max(50, int(0.2 * min(c_w, c_h)))
        else:
            step = int(step_px)
        # move offsets (note: offsets are in canvas pixels)
        self.offset_x += dx_dir * step
        self.offset_y += dy_dir * step
        # preserve offset so we don't recenter
        self._preserve_offset = True
        self._render_image_on_canvas()


    # ---------- Validation helpers ----------
    def validate_all_predictions(self):
        """Move all predictions to GT for current image."""
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        preds = list(self.bboxes_pred.get(img_path, []))
        if not preds:
            messagebox.showinfo("No Predictions", "No predictions to validate for this image.")
            return
        self.bboxes_gt.setdefault(img_path, []).extend(preds)
        self.bboxes_pred[img_path] = []
        self._render_image_on_canvas()
        messagebox.showinfo("Validated", f"Moved {len(preds)} predictions to GT.")

    def validate_all_extra_to_gt(self):
        """Promote extra predicted boxes (unmatched preds) into GT."""
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        preds = list(self.bboxes_pred.get(img_path, []))
        gts = self.bboxes_gt.get(img_path, [])
        extra_preds = []
        for p in preds:
            matched = False
            for g in gts:
                if calculate_iou(p.as_tuple(), g.as_tuple()) > 0.5:
                    matched = True; break
            if not matched:
                extra_preds.append(p)
        if not extra_preds:
            messagebox.showinfo("Nothing to Validate", "No extra predictions to validate.")
            return
        for p in extra_preds:
            if p in self.bboxes_pred.get(img_path, []):
                try: self.bboxes_pred[img_path].remove(p)
                except: pass
            self.bboxes_gt.setdefault(img_path, []).append(p)
        if self.selected_bbox in extra_preds:
            self.selected_bbox = None
        self._render_image_on_canvas()
        messagebox.showinfo("Validated", f"Moved {len(extra_preds)} extra prediction(s) to GT.")

    def validate_selected_prediction(self):
        if not self.selected_bbox:
            messagebox.showwarning("No Selection", "No bbox selected.")
            return
        img_path = self.image_files[self.current_index]
        if self.selected_bbox not in self.bboxes_pred.get(img_path, []):
            messagebox.showwarning("No Selection", "Selected bbox is not a prediction.")
            return
        self.bboxes_pred[img_path].remove(self.selected_bbox)
        self.bboxes_gt.setdefault(img_path, []).append(self.selected_bbox)
        self.selected_bbox = None
        self._render_image_on_canvas()


    # ---------- Save / delete ----------
    def save_current_annotations(self):
        if not self.label_folder:
            messagebox.showwarning("No Labels Folder", "Select a folder for labels first.")
            return
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.label_folder, fname + ".txt")
        bboxes = self.bboxes_gt.get(img_path, [])
        lines = []
        for bb in bboxes:
            nx, ny, nw, nh = bb.normalize(self.img_w, self.img_h)
            nx = min(1.0, max(0.0, nx))
            ny = min(1.0, max(0.0, ny))
            nw = min(1.0, max(0.0, nw))
            nh = min(1.0, max(0.0, nh))
            lines.append(f"{bb.cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
        try:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            # suppress popup printing if user prefers (we won't print)
            # show a small info
            messagebox.showinfo("Saved", f"Annotations saved:\n{txt_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def delete_current_image_and_label(self):
        if not self.image_files or self.current_index < 0:
            return
        img_path = self.image_files[self.current_index]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.label_folder, fname + ".txt")
        pred_txt_path = os.path.join(self.prediction_folder, fname + ".txt")
        try:
            os.remove(img_path)
            if os.path.exists(txt_path):
                os.remove(txt_path)
            if os.path.exists(pred_txt_path):
                os.remove(pred_txt_path)
            self.load_image_list()
            # reload annotations for remaining images (load_image_list resets bbox dicts)
            try:
                self.load_annotations_for_all_images()
            except Exception:
                pass
            self.current_index = min(self.current_index, len(self.image_files) - 1)
            self.show_image_at_index()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete: {e}")

    def delete_all_gt_for_current(self):
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        if not self.bboxes_gt.get(img_path):
            messagebox.showinfo("No GT", "No GT annotations to delete for current image.")
            return
        if not messagebox.askyesno("Confirm", "Delete all GT annotations for current image?"):
            return
        self.bboxes_gt[img_path] = []
        # optionally remove label file on disk:
        if self.label_folder:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.label_folder, fname + ".txt")
            try:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            except Exception:
                pass
        self._render_image_on_canvas()

    # ---------- Navigation ----------
    def next_image(self):
        if not self.image_files:
            return
        self.current_index = min(len(self.image_files) - 1, self.current_index + 1)
        self.show_image_at_index()

    def prev_image(self):
        if not self.image_files:
            return
        self.current_index = max(0, self.current_index - 1)
        self.show_image_at_index()

    def _on_slider_move(self, value):
        if not self.image_files:
            return
        try:
            idx = int(float(value))
            idx = max(0, min(idx, len(self.image_files) - 1))
            if idx != self.current_index:
                self.current_index = idx
                self.show_image_at_index()
        except Exception:
            pass

    def update_mode_buttons_look(self):
        for mk, btn in self.mode_buttons.items():
            cfg = MODES.get(mk, {})
            if self.current_mode.get() == mk:
                btn.configure(bg=cfg.get("color", "#88FF88"), relief=tk.SUNKEN)
            else:
                btn.configure(bg="#DDDDDD", relief=tk.RAISED)

    def set_mode(self, mode_key):
        if self.current_mode.get() == mode_key:
            self.current_mode.set("none")
        else:
            self.current_mode.set(mode_key)
        self.update_mode_buttons_look()

    # ---------- helpers ----------
    def _on_tool_change(self, tool_name):
        self.selected_tool.set(tool_name)
        self._rebuild_tool_panel()

    # ---------- slider helpers ----------
    def update_progress(self):
        if not self.image_files:
            self.slider_var.set(0)
            return
        self.slider_var.set(self.current_index)

# ---------- Run ----------
if __name__ == "__main__":
    app = YoloEditorApp()
    app.mainloop()
