from manim import *


class SampleVideo(Scene):
    def construct(self) -> None:
        title = Text("Manim Sample", font_size=54, weight=BOLD)
        subtitle = Text("Shapes, motion, and a plotted curve", font_size=28)
        subtitle.next_to(title, DOWN)

        self.play(FadeIn(title, shift=UP * 0.4), FadeIn(subtitle, shift=UP * 0.2))
        self.wait(0.5)
        self.play(title.animate.to_edge(UP), FadeOut(subtitle, shift=UP * 0.2))

        circle = Circle(radius=1.0, color=BLUE).shift(LEFT * 3 + DOWN * 0.5)
        square = Square(side_length=1.6, color=GREEN).shift(DOWN * 0.5)
        triangle = Triangle(color=ORANGE).scale(1.1).shift(RIGHT * 3 + DOWN * 0.4)

        labels = VGroup(
            Text("Circle", font_size=24).next_to(circle, DOWN),
            Text("Square", font_size=24).next_to(square, DOWN),
            Text("Triangle", font_size=24).next_to(triangle, DOWN),
        )

        self.play(
            LaggedStart(
                Create(circle),
                Create(square),
                Create(triangle),
                lag_ratio=0.2,
            )
        )
        self.play(LaggedStart(*(FadeIn(label, shift=UP * 0.15) for label in labels), lag_ratio=0.15))
        self.wait(0.5)

        self.play(
            circle.animate.set_fill(BLUE_E, opacity=0.7).rotate(PI / 2),
            square.animate.set_fill(GREEN_E, opacity=0.7).rotate(PI / 4),
            triangle.animate.set_fill(ORANGE, opacity=0.7).rotate(-PI / 3),
        )
        self.wait(0.5)
        self.play(FadeOut(VGroup(circle, square, triangle, labels)))

        axes = Axes(
            x_range=[0, 4 * PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=10,
            y_length=4,
            axis_config={"include_tip": False},
        ).shift(DOWN * 0.4)
        graph = axes.plot(lambda x: 0.9 * np.sin(x), color=YELLOW)
        graph_label = Text("y = 0.9 sin(x)", font_size=26, color=YELLOW)
        graph_label.next_to(axes, DOWN)
        dot = Dot(color=RED).move_to(graph.get_start())

        self.play(Create(axes))
        self.play(Create(graph), FadeIn(graph_label, shift=UP * 0.15))
        self.play(MoveAlongPath(dot, graph), run_time=3, rate_func=linear)
        self.wait(0.5)
        self.play(FadeOut(VGroup(axes, graph, graph_label, dot, title)))
