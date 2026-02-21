import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageProcessorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Processor")
        self.root.configure(bg="#1e1e2e")
        self.root.minsize(950, 650)

        self.original_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None

        self._build_ui()

    # UI
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6, font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#1e1e2e", foreground="white")
        style.configure("TLabelframe.Label", background="#1e1e2e", foreground="white", font=("Segoe UI", 11, "bold"))
        style.configure("TScale", background="#1e1e2e")
        style.configure("Header.TLabel", background="#1e1e2e", foreground="white", font=("Segoe UI", 14, "bold"))

        # top bar
        top = tk.Frame(self.root, bg="#1e1e2e")
        top.pack(fill="x", padx=10, pady=(10, 5))

        ttk.Button(top, text=" Upload Image", command=self._upload_image).pack(side="left", padx=5)
        ttk.Button(top, text=" Reset", command=self._reset_image).pack(side="left", padx=5)
        ttk.Button(top, text=" Save", command=self._save_image).pack(side="left", padx=5)

        # main area, canvas left, actions on the right
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=10, pady=5)

        # image preview frame
        canvas_frame = tk.Frame(main, bg="#2a2a3c", bd=2, relief="sunken")
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.canvas = tk.Canvas(canvas_frame, bg="#2a2a3c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._display_image())

        # scrollable side bar 
        ctrl_outer = tk.Frame(main, bg="#1e1e2e", width=280)
        ctrl_outer.pack(side="right", fill="y")
        ctrl_outer.pack_propagate(False)

        ctrl_canvas = tk.Canvas(ctrl_outer, bg="#1e1e2e", highlightthickness=0, width=260)
        scrollbar = ttk.Scrollbar(ctrl_outer, orient="vertical", command=ctrl_canvas.yview)
        self.ctrl_frame = tk.Frame(ctrl_canvas, bg="#1e1e2e")

        self.ctrl_frame.bind(
            "<Configure>",
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")),
        )
        ctrl_canvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)

        ctrl_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # mouse scroll wheel event
        def _on_mousewheel(event):
            ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_controls()

    def _build_controls(self):
        parent = self.ctrl_frame

        # grayscale
        gray_frame = ttk.LabelFrame(parent, text="Grayscale")
        gray_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(gray_frame, text="Convert to Grayscale", command=self._apply_grayscale).pack(fill="x", padx=5, pady=5)

        # segementation / thresholding
        thresh_frame = ttk.LabelFrame(parent, text="Segmentation / Threshold")
        thresh_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(thresh_frame, text="Threshold value", bg="#1e1e2e", fg="white").pack(anchor="w", padx=5)
        self.thresh_slider = tk.Scale(
            thresh_frame, from_=0, to=255, orient="horizontal",
            bg="#1e1e2e", fg="white", highlightthickness=0,
        )
        self.thresh_slider.set(127)
        self.thresh_slider.pack(fill="x", padx=5)

        ttk.Button(thresh_frame, text="Thresh Binary", command=self._apply_thresh_binary).pack(fill="x", padx=5, pady=5)
        ttk.Button(thresh_frame, text="Thresh Binary Inv", command=self._apply_thresh_binary_inv).pack(fill="x", padx=5, pady=(0, 5))
        ttk.Button(thresh_frame, text="Thresh Truncate", command=self._apply_thresh_trunc).pack(fill="x", padx=5, pady=(0, 5))
        ttk.Button(thresh_frame, text="Thresh To Zero", command=self._apply_thresh_tozero).pack(fill="x", padx=5, pady=(0, 5))
        ttk.Button(thresh_frame, text="Otsu's Threshold", command=self._apply_thresh_otsu).pack(fill="x", padx=5, pady=(0, 5))

        # adaptive thresholding
        adapt_frame = ttk.LabelFrame(parent, text="Adaptive Threshold")
        adapt_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(adapt_frame, text="Block size (odd ≥ 3)", bg="#1e1e2e", fg="white").pack(anchor="w", padx=5)
        self.adapt_block_slider = tk.Scale(
            adapt_frame, from_=3, to=51, orient="horizontal",
            bg="#1e1e2e", fg="white", highlightthickness=0, resolution=2,
        )
        self.adapt_block_slider.set(11)
        self.adapt_block_slider.pack(fill="x", padx=5)

        tk.Label(adapt_frame, text="C constant", bg="#1e1e2e", fg="white").pack(anchor="w", padx=5)
        self.adapt_c_slider = tk.Scale(
            adapt_frame, from_=0, to=30, orient="horizontal",
            bg="#1e1e2e", fg="white", highlightthickness=0,
        )
        self.adapt_c_slider.set(2)
        self.adapt_c_slider.pack(fill="x", padx=5)

        ttk.Button(adapt_frame, text="Adaptive Gaussian", command=self._apply_adaptive_gaussian).pack(fill="x", padx=5, pady=5)
        ttk.Button(adapt_frame, text="Adaptive Mean", command=self._apply_adaptive_mean).pack(fill="x", padx=5, pady=(0, 5))

        # edge detection
        edge_frame = ttk.LabelFrame(parent, text="Edge Detection")
        edge_frame.pack(fill="x", padx=5, pady=5)
        ttk.Button(edge_frame, text="Canny Edge", command=self._apply_canny).pack(fill="x", padx=5, pady=5)
        ttk.Button(edge_frame, text="Sobel Edge", command=self._apply_sobel).pack(fill="x", padx=5, pady=(0, 5))
        ttk.Button(edge_frame, text="Laplacian Edge", command=self._apply_laplacian).pack(fill="x", padx=5, pady=(0, 5))

    # file actions
    def _upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        self.original_image = img.copy()
        self.processed_image = img.copy()
        self._display_image()

    def _reset_image(self):
        if self.original_image is None:
            return
        self.processed_image = self.original_image.copy()
        self._display_image()

    def _save_image(self):
        if self.processed_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if path:
            cv2.imwrite(path, self.processed_image)

    # display helper
    def _display_image(self):
        if self.processed_image is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        img = self.processed_image.copy()
        # convert BGR → RGB for PIL
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(img_resized)
        self._tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, anchor="center", image=self._tk_image)

    # augmentations
    def _ensure_bgr(self) -> np.ndarray:
        img = self.processed_image
        if img is None:
            return None
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _ensure_gray(self) -> np.ndarray:
        img = self.processed_image
        if img is None:
            return None
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # grayscale
    def _apply_grayscale(self):
        if self.processed_image is None:
            return
        self.processed_image = self._ensure_gray()
        self._display_image()

    # simple thresholds
    def _apply_threshold(self, thresh_type, use_otsu=False):
        gray = self._ensure_gray()
        if gray is None:
            return
        val = self.thresh_slider.get()
        flags = thresh_type | (cv2.THRESH_OTSU if use_otsu else 0)
        _, result = cv2.threshold(gray, val if not use_otsu else 0, 255, flags)
        self.processed_image = result
        self._display_image()

    def _apply_thresh_binary(self):
        self._apply_threshold(cv2.THRESH_BINARY)

    def _apply_thresh_binary_inv(self):
        self._apply_threshold(cv2.THRESH_BINARY_INV)

    def _apply_thresh_trunc(self):
        self._apply_threshold(cv2.THRESH_TRUNC)

    def _apply_thresh_tozero(self):
        self._apply_threshold(cv2.THRESH_TOZERO)

    def _apply_thresh_otsu(self):
        self._apply_threshold(cv2.THRESH_BINARY, use_otsu=True)

    # adaptive thresholds
    def _apply_adaptive(self, method):
        gray = self._ensure_gray()
        if gray is None:
            return
        block = self.adapt_block_slider.get()
        block = max(3, block if block % 2 == 1 else block + 1)
        c = self.adapt_c_slider.get()
        result = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, block, c)
        self.processed_image = result
        self._display_image()

    def _apply_adaptive_gaussian(self):
        self._apply_adaptive(cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    def _apply_adaptive_mean(self):
        self._apply_adaptive(cv2.ADAPTIVE_THRESH_MEAN_C)
 
    # edge detection 
    def _apply_canny(self):
        gray = self._ensure_gray()
        if gray is None:
            return
        self.processed_image = cv2.Canny(gray, 50, 150)
        self._display_image()

    def _apply_sobel(self):
        gray = self._ensure_gray()
        if gray is None:
            return
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        self.processed_image = np.clip(magnitude, 0, 255).astype(np.uint8)
        self._display_image()

    def _apply_laplacian(self):
        gray = self._ensure_gray()
        if gray is None:
            return
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        self.processed_image = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
        self._display_image()


def main():
    root = tk.Tk()
    root.geometry("1100x700")
    ImageProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
