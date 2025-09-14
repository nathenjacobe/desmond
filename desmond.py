import tkinter as tk
from math import sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, sqrt, cbrt, log, pi, e, exp, floor, ceil
from cmath import sin as csin, sqrt as csqrt, exp as cexp

EPSILON = 1e-07
LINE_EPSILON = 0.1
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

SAFE_GLOBALS = {
    "sin": sin, 
    "cos": cos, 
    "tan": tan, 
    "csc": lambda x: 1/sin(x),
    "sec": lambda x: 1/cos(x),
    "cot": lambda x: 1/tan(x),
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "sinh": sinh,
    "cosh": cosh,
    "tanh": tanh,
    "sqrt": sqrt, 
    "abs": abs,
    "floor": floor,
    "ceil": ceil,
    "round": round,
    "cbrt": cbrt,
    "binomial": lambda x, n, p: choose(n, x) * p**x * (1-p)**(n-x),
    "normal": lambda x, m, s: (1 / (sqrt(2 * pi * s * s))) * exp(-((x-m)**2 / (2 * s * s))),
    "poisson": lambda x, l: None if x < 0 else (l**x * exp(-l)) / factorial(x),
    "erf": lambda x: tanh(x * pi / sqrt(6)),
    "factorial": factorial,
    "log": log,
    "pi": pi, 
    "e": e,
    "gamma": 0.57721566490153286060651209008240243104215933593992
}


class Desmond:
    def __init__(self, root):
        self.root = root
        self.root.title("desmond")
        self.root.geometry("800x850")

        self.width = 800
        self.height = 800
        self.range_x = (-10, 10)
        self.range_y = (-10, 10)
        self.grid_step = 1

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.main_frame, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, pady=5)

        tk.Label(control_frame, text="enter graph to plot").pack(side=tk.LEFT, padx=(10, 5))
        self.entry = tk.Entry(control_frame, width=50)
        self.entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.entry.insert(0, "")

        self.entry.bind('<Control-a>', self._select_all)

        self.plot_button = tk.Button(control_frame, text="Plot", command=self.plot_dispatcher)
        self.plot_button.pack(side=tk.LEFT, padx=5)

        self.draw_axes()

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

    def draw_axes(self):
        self.canvas.delete("all")
        self.canvas.configure(bg="white")

        for i in range(int(self.range_x[0]), int(self.range_x[1]) + 1, self.grid_step):
            if i == 0: continue
            x_canvas, _ = self.to_canvas_coords(i, 0)
            self.canvas.create_line(x_canvas, 0, x_canvas, self.height, fill="#f0f0f0")
            self.canvas.create_text(x_canvas, self.to_canvas_coords(0, 0)[1] + 10, text=str(i), anchor=tk.N, fill="gray")

        for i in range(int(self.range_y[0]), int(self.range_y[1]) + 1, self.grid_step):
            if i == 0: continue
            _, y_canvas = self.to_canvas_coords(0, i)
            self.canvas.create_line(0, y_canvas, self.width, y_canvas, fill="#f0f0f0")
            if i != 0:
                self.canvas.create_text(self.to_canvas_coords(0, 0)[0] - 10, y_canvas, text=str(i), anchor=tk.E, fill="gray")

        x_origin, y_origin = self.to_canvas_coords(0, 0)
        self.canvas.create_line(0, y_origin, self.width, y_origin, fill="black", width=2)
        self.canvas.create_line(x_origin, 0, x_origin, self.height, fill="black", width=2)
        self.canvas.create_text(x_origin + 10, y_origin + 10, text="0", anchor=tk.NW, fill="black")

    def plot_dispatcher(self):
        self.draw_axes()

        raw_func_str = self.entry.get()
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
                self.show_error("must only have 1 equals sign")
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

        if is_explicit:
            self.plot_explicit_function(expression)
        else:
            self.plot_implicit_function(expression)

    def plot_explicit_function(self, expression_str):
        try:
            code = compile(expression_str, '<string>', 'eval')
        except SyntaxError:
            self.show_error("invalid syntax. did you forget to close your brackets?")
            return

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
                         self.canvas.create_line(last_c_coords, (cx, cy), fill="blue", width=2)
                
                last_c_coords = (cx, cy)
                
            except ZeroDivisionError:
                last_c_coords = None
                c_asymptote_x, _ = self.to_canvas_coords(x, 0)
                self.canvas.create_line(
                    c_asymptote_x, 0, c_asymptote_x, self.height, 
                    fill="red", dash=(4, 2), width=1.5
                )
            except (ValueError, OverflowError, TypeError):
                last_c_coords = None

    def plot_implicit_function(self, expression_str):
        try:
            code = compile(expression_str, '<string>', 'eval')
        except SyntaxError:
            self.show_error("invalid syntax. did you forget to close your brackets?")
            return

        img = tk.PhotoImage(width=self.width, height=self.height)
        self.canvas.create_image((self.width / 2, self.height / 2), image=img, state="normal")
        
        line_color = "#0000ff"

        for px in range(self.width):
            for py in range(self.height):
                x, y = self.to_math_coords(px, py)
                try:
                    result = eval(code, SAFE_GLOBALS, {"x": x, "y": y})
                    if abs(result) < LINE_EPSILON:
                        img.put(line_color, (px, py))
                except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                    continue

        self.canvas.image = img

    def show_error(self, message):
        self.canvas.create_text(
            self.width / 2, 20, text=message, fill="red", font=("Arial", 12)
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = Desmond(root)
    root.mainloop()