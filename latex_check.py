from manim import *


class LatexCheck(Scene):
    def construct(self) -> None:
        expr = Tex(r"Manim + \LaTeX")
        formula = MathTex(r"\int_0^\pi \sin(x)\,dx = 2")
        formula.next_to(expr, DOWN, buff=0.6)

        self.play(Write(expr))
        self.play(Write(formula))
        self.wait(0.5)
