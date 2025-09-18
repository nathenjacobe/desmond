import tkinter as tk
from tkinter import ttk
from math import sin, cos, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, sqrt, cbrt, log, pi, e, exp, floor, ceil
from cmath import sin as csin, sqrt as csqrt, exp as cexp
from collections import defaultdict
import random

EPSILON = 1e-07
g = 7
n = 9
p = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
]

def is_close_to_int(x):
    return abs(x - round(x)) <= EPSILON

def drop_imag(z):
    if abs(z.imag) <= EPSILON:
        z = z.real
    return z

# lanczos approx; from wikipedia: https://en.wikipedia.org/wiki/Lanczos_approximation
def gamma(z):
    z = complex(z)
    if z.real < 0.5:
        y = pi / (csin(pi * z) * gamma(1 - z))
    else:
        z -= 1
        x = p[0]
        for i in range(1, len(p)):
            x += p[i] / (z + i)
        t = z + g + 0.5
        y = csqrt(2 * pi) * t ** (z + 0.5) * cexp(-t) * x
    return drop_imag(y)

def factorial(x): return gamma(x+1)

def choose(n, k):
    if k < 0: raise OverflowError

    k_round = int(round(k))
    if abs(k - k_round) <= EPSILON:
        k = k_round
    else:
        return 0

    if k > n:
        raise OverflowError
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    numerator = 1
    denominator = 1
    for i in range(k):
        numerator *= (n - i)
        denominator *= (i + 1)
    return numerator // denominator

def lerp(a, b, t):
    return a + (b - a) * t

SAFE_GLOBALS = {
    "sin": sin, 
    "cos": cos, 
    "tan": lambda x: sin(x)/cos(x), 

    "csc": lambda x: 1/sin(x),
    "sec": lambda x: 1/cos(x),
    "cot": lambda x: cos(x)/sin(x),

    "asin": asin,
    "acos": acos,
    "atan": atan,

    "acsc": lambda x: asin(1/x),
    "asec": lambda x: acos(1/x),
    "acot": lambda x: atan(1/x),

    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,

    "csch": lambda x: 1/sinh(x),
    "sech": lambda x: 1/cosh(x),
    "coth": lambda x: 1/tanh(x),

    "asinh": asinh,
    "acosh": acosh,
    "atanh": atanh,

    "acsch": lambda x: asinh(1/x),
    "asech": lambda x: acosh(1/x),
    "acoth": lambda x: atanh(1/x),

    "sqrt": sqrt, 
    "cbrt": cbrt,

    "abs": abs,
    "floor": floor,
    "ceil": ceil,
    "round": round,
    "min": min,
    "max": max,
    "clamp": lambda x, a, b: max(a, min(x, b)),
    "lerp": lerp,

    "binomial": lambda x, n, p: choose(n, x) * p**x * (1-p)**(n-x),
    "normal": lambda x, m, s: (1 / (sqrt(2 * pi * s * s))) * exp(-((x-m)**2 / (2 * s * s))),
    "poisson": lambda x, l: None if x < 0 else (l**x * exp(-l)) / factorial(x),
    "geo": lambda x, p: None if x < 0 else (1-p)**(x-1)*p,

    "erf": lambda x: tanh(x * pi / sqrt(6)),
    "factorial": factorial,
    "log": log,

    "pi": pi, 
    "e": e,
    "gamma": 0.57721566490153286060651209008240243104215933593992
}

def interpolate(pa, pb, va, vb):
    # basic lerp 
    denom = (va - vb)
    if abs(denom) < EPSILON:
        t = 0.5
    else:
        t = va / denom
    t = max(0.0, min(1.0, t))
    x = lerp(pa[0], pb[0], t)
    y = lerp(pa[1], pb[1], t)
    return (x, y)



class Graph:
    def __init__(self, expression, color, is_explicit):
        self.expression = expression
        self.color = color
        self.is_explicit = is_explicit
        self.canvas_items = []

class Desmond:
    def __init__(self, root):
        self.root = root
        self.root.title("desmond")
        self.root.geometry("1100x800")

        self.width = 800
        self.height = 800
        self.range_x = (-10, 10)
        self.range_y = (-10, 10)
        self.grid_step = 1

        self.graphs = {} 
        self.next_graph_id = 1
        
        self.colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

        self.setup_ui()
        self.draw_axes()

    def setup_ui(self):
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = tk.Frame(main_container, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = tk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_panel, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        input_frame = tk.Frame(left_panel)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(input_frame, text="enter your graph:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.entry = tk.Entry(input_frame, width=25)
        self.entry.pack(fill=tk.X, pady=(2, 5))

        # thanks stack overflow!!1!
        self.entry.bind('<Return>', self.add_graph)
        self.entry.bind('<Control-a>', self._select_all)

        color_frame = tk.Frame(input_frame)
        color_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Label(color_frame, text="graph color:").pack(side=tk.LEFT)
        self.color_var = tk.StringVar(value="blue")
        color_dropdown = ttk.Combobox(color_frame, textvariable=self.color_var, 
                                     values=self.colors, width=8, state="readonly")
        color_dropdown.pack(side=tk.RIGHT)

        button_frame = tk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.add_button = tk.Button(button_frame, text="+", command=self.add_graph, bg="#4CAF50", fg="white")
        self.add_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.clear_button = tk.Button(button_frame, text="AC", command=self.clear_all_graphs, bg="#f44336", fg="white")
        self.clear_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        tk.Label(left_panel, text="graphs:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        list_frame = tk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas_frame = tk.Canvas(list_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas_frame.yview)
        self.scrollable_frame = tk.Frame(canvas_frame)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda _: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )

        canvas_frame.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)

        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _select_all(self, event):
        widget = event.widget
        widget.select_range(0, 'end')
        widget.icursor('end')
        return 'break'

    def add_implicit_multiplication(self, expr):
        expr = expr.replace(" ", "")
        new_expr = []
        for i, char in enumerate(expr):
            new_expr.append(char)
            if i < len(expr) - 1:
                next_char = expr[i+1]
                if (char.isdigit() or char == ')' or char in 'xy') and \
                   (next_char.isalpha() or next_char == '('):
                    if not (char == 'e' and (i > 0 and expr[i-1].isdigit())):
                         new_expr.append('*')
                elif char in 'xy' and next_char.isdigit():
                    new_expr.append('*')
        return "".join(new_expr)

    def to_canvas_coords(self, x, y):
        x_canvas = (x - self.range_x[0]) / (self.range_x[1] - self.range_x[0]) * self.width
        y_canvas = self.height - (y - self.range_y[0]) / (self.range_y[1] - self.range_y[0]) * self.height
        return x_canvas, y_canvas

    def to_math_coords(self, x_canvas, y_canvas):
        x = (x_canvas / self.width) * (self.range_x[1] - self.range_x[0]) + self.range_x[0]
        y = ((self.height - y_canvas) / self.height) * (self.range_y[1] - self.range_y[0]) + self.range_y[0]
        return x, y
    
    def key_of(self, pt):
        cx, cy = self.to_canvas_coords(pt[0], pt[1])
        return (int(round(cx)), int(round(cy)))

    def draw_axes(self):
        self.canvas.delete("axes")
        self.canvas.delete("grid")
        
        for i in range(int(self.range_x[0]), int(self.range_x[1]) + 1, self.grid_step):
            if i == 0: continue
            x_canvas, _ = self.to_canvas_coords(i, 0)
            self.canvas.create_line(x_canvas, 0, x_canvas, self.height, fill="#f0f0f0", tags="grid")
            self.canvas.create_text(x_canvas, self.to_canvas_coords(0, 0)[1] + 10, text=str(i), anchor=tk.N, fill="gray", tags="grid")

        for i in range(int(self.range_y[0]), int(self.range_y[1]) + 1, self.grid_step):
            if i == 0: continue
            _, y_canvas = self.to_canvas_coords(0, i)
            self.canvas.create_line(0, y_canvas, self.width, y_canvas, fill="#f0f0f0", tags="grid")
            if i != 0:
                self.canvas.create_text(self.to_canvas_coords(0, 0)[0] - 10, y_canvas, text=str(i), anchor=tk.E, fill="gray", tags="grid")

        x_origin, y_origin = self.to_canvas_coords(0, 0)
        self.canvas.create_line(0, y_origin, self.width, y_origin, fill="black", width=2, tags="axes")
        self.canvas.create_line(x_origin, 0, x_origin, self.height, fill="black", width=2, tags="axes")
        self.canvas.create_text(x_origin + 10, y_origin + 10, text="0", anchor=tk.NW, fill="black", tags="axes")

    def add_graph(self):
        raw_func_str = self.entry.get().strip()
        if not raw_func_str:
            return

        func_str = self.add_implicit_multiplication(raw_func_str)
        func_str = func_str.replace('^', '**')

        is_explicit = False
        expression = ""

        if '=' not in func_str:
            if 'y' not in func_str:
                is_explicit = True
                expression = func_str
            else:
                is_explicit = False
                expression = func_str
        else:
            parts = func_str.split('=')
            if len(parts) != 2:
                self.show_error("max 1 equals sign")
                return

            lhs, rhs = parts[0].strip(), parts[1].strip()

            if (lhs == 'y' and 'y' not in rhs):
                is_explicit = True
                expression = rhs
            elif (rhs == 'y' and 'y' not in lhs):
                is_explicit = True
                expression = lhs
            else:
                is_explicit = False
                expression = f"({lhs}) - ({rhs})"

        color = self.color_var.get()
        graph = Graph(expression, color, is_explicit)
        graph_id = self.next_graph_id
        self.graphs[graph_id] = graph
        self.next_graph_id += 1

        self.plot_graph(graph)
        
        self.add_graph_to_list(graph_id, raw_func_str, color)

        self.entry.delete(0, tk.END)
        self.color_var.set(random.choice(self.colors))

    def plot_graph(self, graph):
        self.plot_explicit_function(graph) if graph.is_explicit else self.plot_implicit_function(graph)

    def plot_explicit_function(self, graph):
        try:
            code = compile(graph.expression, '<string>', 'eval')
        except SyntaxError:
            self.show_error("invalid syntax; did you forget to close your brackets?")

        last_c_coords = None
        for px in range(self.width):
            x, _ = self.to_math_coords(px, 0)
            try:
                y = eval(code, SAFE_GLOBALS, {"x": x})

                if not isinstance(y, (int, float)):
                    last_c_coords = None
                    continue

                cx, cy = self.to_canvas_coords(x, y)

                if last_c_coords is not None:
                    if abs(cy - last_c_coords[1]) < self.height:
                        item_id = self.canvas.create_line(last_c_coords, (cx, cy), fill=graph.color, width=2)
                        graph.canvas_items.append(item_id)
                
                last_c_coords = (cx, cy)
                
            except ZeroDivisionError:
                last_c_coords = None
                c_asymptote_x, _ = self.to_canvas_coords(x, 0)
                item_id = self.canvas.create_line(
                    c_asymptote_x, 0, c_asymptote_x, self.height, 
                    fill="red", dash=(4, 2), width=1.5
                )
                graph.canvas_items.append(item_id)
            except (ValueError, OverflowError, TypeError):
                last_c_coords = None

    def plot_implicit_function(self, graph):
        try:
            code = compile(graph.expression, "<string>", "eval")
        except SyntaxError:
            self.show_error("invalid syntax; did you forget to close your brackets?")

        # the idea for this is to use the 2d version of marching cubes, marching squares
        # cf: https://en.wikipedia.org/wiki/Marching_squares. the diagrams on there are really helpful for visualising this
        # There are only 16 cases and they can all be resolved to generate segment(s).
        # a problem arises if you have a saddle point, which is ambiguous. for this case, i simply check the center and go from there
        # to form the curve, it stitches all the segments together, moving backwards and forwards...
        # ... to build a series of connected segments, called a polyline


        # sample interval; keeping at 5 for, now but in theory lower = better but less performant and more susceptible to noise?
        # something something Nyquist something something
        cell_px = 5
        cols = max(4, self.width // cell_px)
        rows = max(4, self.height // cell_px)

        # sample and cache
        grid_vals = [[None]*(cols+1) for _ in range(rows+1)]
        pts = [[(0,0)]*(cols+1) for _ in range(rows+1)]
        for j in range(rows+1):
            cy = round(j * (self.height-1) / rows)
            for i in range(cols+1):
                cx = round(i * (self.width-1) / cols)
                x, y = self.to_math_coords(cx, cy)
                pts[j][i] = (x, y)
                try:
                    v = eval(code, SAFE_GLOBALS, {"x": x, "y": y})
                    grid_vals[j][i] = float(v) if isinstance(v, (int, float)) else None
                except Exception:
                    grid_vals[j][i] = None

        segments = []

        for j in range(rows):
            for i in range(cols):
                v00 = grid_vals[j][i]
                v10 = grid_vals[j][i+1]
                v11 = grid_vals[j+1][i+1]
                v01 = grid_vals[j+1][i]
                if None in (v00, v10, v11, v01):
                    continue
                p00 = pts[j][i]; p10 = pts[j][i+1]; p11 = pts[j+1][i+1]; p01 = pts[j+1][i]

                corners = [v00, v10, v11, v01]
                pattern = 0
                for k, val in enumerate(corners):
                    if val < 0: pattern |= (1 << k)

                # for all 16 edges decide segments using interpolation
                inter = []
                edges = [ (p00,p10,v00,v10), (p10,p11,v10,v11),
                          (p11,p01,v11,v01), (p01,p00,v01,v00) ]
                for (pa,pb,va,vb) in edges:
                    if va * vb < 0:
                        inter.append(interpolate(pa,pb,va,vb))
                    elif abs(va) <= EPSILON:
                        inter.append(pa)
                    elif abs(vb) <= EPSILON:
                        inter.append(pb)

                # 2 intersections = make a segment
                if len(inter) == 2:
                    segments.append((inter[0], inter[1]))
                # 4 intersections = get the center and use it to determine where to make the 2 segments
                elif len(inter) == 4:
                    cx = (p00[0]+p10[0]+p11[0]+p01[0])/4.0
                    cy = (p00[1]+p10[1]+p11[1]+p01[1])/4.0
                    try:
                        center_val = eval(code, SAFE_GLOBALS, {"x":cx, "y":cy})
                        center_pos = center_val > 0
                    except Exception:
                        center_pos = True
                    if center_pos:
                        segments.append((inter[0], inter[1]))
                        segments.append((inter[2], inter[3]))
                    else:
                        segments.append((inter[1], inter[2]))
                        segments.append((inter[3], inter[0]))

        endpoint_map = defaultdict(list)
        seg_used = [False]*len(segments)
        for idx, (a,b) in enumerate(segments):
            endpoint_map[self.key_of(a)].append((idx, 'a'))
            endpoint_map[self.key_of(b)].append((idx, 'b'))

        polylines = []
        # this is kinda messy but it gets the job done
        # the idea is to collate all the segments and connect them if they are sufficiently close
        # then feed it into create_line to hopefully render a beautiful continuous curve.
        for i_seg, seg in enumerate(segments):
            if seg_used[i_seg]:
                continue
            a,b = seg
            seg_used[i_seg] = True
            poly = [a, b]

            # forward
            cur = b
            while True:
                k = self.key_of(cur)
                found = False
                for (sid, _) in endpoint_map.get(k, []):
                    if seg_used[sid]: continue
                    seg_used[sid] = True
                    sa, sb = segments[sid]
                    nxt = sb if self._points_close(sa, cur) else sa
                    poly.append(nxt)
                    cur = nxt
                    found = True
                    break
                if not found: break


            # backward
            cur = a
            while True:
                k = self.key_of(cur)
                found = False
                for (sid, _) in endpoint_map.get(k, []):
                    if seg_used[sid]: continue
                    seg_used[sid] = True
                    sa, sb = segments[sid]
                    nxt = sb if self._points_close(sa, cur) else sa
                    poly.insert(0, nxt)
                    cur = nxt
                    found = True
                    break
                if not found: break

            if len(poly) >= 2:
                polylines.append(poly)

        # draw all gathered polylines
        for poly in polylines:
            coords = []
            for (x,y) in poly:
                cx, cy = self.to_canvas_coords(x, y)
                coords.extend((cx, cy))
            item_id = self.canvas.create_line(*coords, fill=graph.color, width=2)
            graph.canvas_items.append(item_id)

    def _points_close(self, p1, p2, tol_px=3):
        c1 = self.to_canvas_coords(p1[0], p1[1])
        c2 = self.to_canvas_coords(p2[0], p2[1])
        dx = c1[0] - c2[0]; dy = c1[1] - c2[1]
        return (dx*dx + dy*dy) <= (tol_px*tol_px)

    def add_graph_to_list(self, graph_id, expression, color):
        frame = tk.Frame(self.scrollable_frame, relief=tk.RAISED, borderwidth=1)
        frame.pack(fill=tk.X, padx=2, pady=1)

        remove_btn = tk.Button(frame, text="×", command=lambda: self.remove_graph(graph_id, frame),
                            font=("Arial", 10, "bold"), width=2, height=1,
                            bg="#ff4444", fg="white", relief=tk.FLAT)
        remove_btn.pack(side=tk.LEFT, padx=5)

        color_canvas = tk.Canvas(frame, width=15, height=15, highlightthickness=0)
        color_canvas.pack(side=tk.LEFT, padx=(5, 5), pady=5)
        color_canvas.create_rectangle(2, 2, 13, 13, fill=color, outline="black")

        expr_label = tk.Label(frame, text=expression[:23] + ("..." if len(expression) > 20 else ""),
                            anchor="w", font=("Arial", 9))
        expr_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    def remove_graph(self, graph_id, frame):
        if graph_id in self.graphs:
            graph = self.graphs[graph_id]
            for item_id in graph.canvas_items:
                self.canvas.delete(item_id)
            
            del self.graphs[graph_id]
        
        frame.destroy()

    def clear_all_graphs(self):
        for graph in self.graphs.values():
            for item_id in graph.canvas_items:
                self.canvas.delete(item_id)
        
        self.graphs.clear()
        
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.draw_axes()

    def show_error(self, message):
        self.canvas.create_text(
            self.width / 2, 20, text=message, fill="red", font=("Arial", 12), tags="error"
        )
        self.root.after(3000, lambda: self.canvas.delete("error"))

if __name__ == "__main__":
    root = tk.Tk()
    app = Desmond(root)
    root.mainloop()