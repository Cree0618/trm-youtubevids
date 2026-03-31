from manim import *


BG = "#08111d"
PANEL = "#0f1c2e"
PANEL_2 = "#13243a"
FG = "#f5f7fb"
MUTED = "#a5b3c7"
SIGN_C = "#7f8c8d"
FP16_C = "#5dade2"
BF16_C = "#58d68d"
FP8_C = "#f5b041"
MXFP8_C = "#af7ac5"
NVFP4_C = "#ec7063"
FONT = "Avenir Next"


class WeightFormatsExplainer(Scene):
    def setup(self):
        self.camera.background_color = BG

    def txt(self, content, font_size=24, color=FG, weight=NORMAL, max_width=None):
        text = Text(content, font=FONT, font_size=font_size, color=color, weight=weight)
        if max_width and text.width > max_width:
            text.scale_to_fit_width(max_width)
        return text

    def panel(self, width, height, color=PANEL, opacity=0.96, stroke=GREY_B):
        return RoundedRectangle(
            corner_radius=0.18,
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=opacity,
            stroke_color=stroke,
            stroke_width=1.6,
        )

    def title_block(self, title, subtitle):
        title_text = self.txt(title, font_size=42, color=FG, weight=BOLD, max_width=10.8)
        subtitle_text = self.txt(subtitle, font_size=24, color=MUTED, max_width=10.6)
        subtitle_text.next_to(title_text, DOWN, buff=0.16)
        panel = self.panel(11.5, 1.65, color="#0a1728", opacity=0.92, stroke="#24384f")
        group = VGroup(panel, title_text, subtitle_text)
        title_text.move_to(panel.get_center() + UP * 0.22)
        subtitle_text.move_to(panel.get_center() + DOWN * 0.38)
        return group

    def bit_bar(self, label, parts, colors, width=4.7):
        total = sum(parts)
        segments = VGroup()
        labels = VGroup()
        cursor = LEFT * (width / 2)
        for part, color in zip(parts, colors):
            w = width * part / total
            rect = Rectangle(width=w, height=0.54, stroke_width=0, fill_color=color, fill_opacity=1)
            rect.move_to(cursor + RIGHT * (w / 2))
            cursor += RIGHT * w
            segments.add(rect)
            labels.add(self.txt(str(part), font_size=18, color=FG, weight=BOLD).move_to(rect.get_center()))
        frame = SurroundingRectangle(segments, buff=0.0, color="#20374c", stroke_width=1.2)
        title = self.txt(label, font_size=24, color=FG, weight=BOLD, max_width=width).next_to(segments, UP, buff=0.16)
        return VGroup(title, frame, segments, labels)

    def format_card(self, name, bits_line, bullets, accent, width=3.6, height=2.65):
        box = self.panel(width, height, color=PANEL_2, opacity=0.98, stroke=accent)
        title = self.txt(name, font_size=27, color=FG, weight=BOLD, max_width=width - 0.4).move_to(box.get_top() + DOWN * 0.34)
        line = self.txt(bits_line, font_size=18, color=accent, max_width=width - 0.4).next_to(title, DOWN, buff=0.14)
        bullet_group = VGroup(*[self.txt(b, font_size=17, color=FG, max_width=width - 0.45) for b in bullets]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12
        )
        bullet_group.next_to(line, DOWN, buff=0.18)
        bullet_group.align_to(box.get_left() + RIGHT * 0.22, LEFT)
        return VGroup(box, title, line, bullet_group)

    def block_grid(self, n, cols, color, side=0.26):
        cells = VGroup(
            *[
                Square(side_length=side, stroke_color=color, stroke_width=1.2, fill_color=color, fill_opacity=0.14)
                for _ in range(n)
            ]
        )
        rows = int(np.ceil(n / cols))
        cells.arrange_in_grid(rows=rows, cols=cols, buff=0.04)
        return cells

    def model_row(self, name, values, value_labels, max_value):
        row_label = self.txt(name, font_size=22, color=FG, weight=BOLD, max_width=1.6)
        labels = ["FP16/BF16", "FP8", "NVFP4 raw"]
        colors = [FP16_C, FP8_C, NVFP4_C]
        bars = VGroup()
        text_labels = VGroup()
        for label, value, value_text, color in zip(labels, values, value_labels, colors):
            track = RoundedRectangle(
                corner_radius=0.08,
                width=3.9,
                height=0.22,
                stroke_width=0,
                fill_color="#18293f",
                fill_opacity=1,
            )
            fill = RoundedRectangle(
                corner_radius=0.08,
                width=max(0.18, 3.9 * value / max_value),
                height=0.22,
                stroke_width=0,
                fill_color=color,
                fill_opacity=1,
            )
            fill.align_to(track, LEFT)
            label_text = self.txt(f"{label}  {value_text}", font_size=18, color=FG, max_width=3.0)
            group = VGroup(track, fill, label_text)
            label_text.next_to(track, RIGHT, buff=0.16)
            bars.add(VGroup(track, fill))
            text_labels.add(label_text)
        stacks = VGroup(*[VGroup(bar, txt).arrange(RIGHT, buff=0.16) for bar, txt in zip(bars, text_labels)]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.15
        )
        row = VGroup(row_label, stacks).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        return row

    def construct(self):
        title = self.title_block(
            "Weight Formats for Inference",
            "Range, precision, scaling, memory footprint",
        )
        title.to_edge(UP, buff=0.35)
        self.play(FadeIn(title, shift=UP * 0.2))
        self.wait(0.4)

        hook = self.txt(
            "Format is not just bit count. It is payload plus tradeoffs.",
            font_size=28,
            color=FG,
            weight=BOLD,
            max_width=10.0,
        )
        hook_panel = self.panel(10.6, 1.0, color=PANEL, stroke="#1e3148")
        hook_group = VGroup(hook_panel, hook)
        hook.move_to(hook_panel.get_center())
        hook_group.next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(hook_group, shift=UP * 0.15))
        self.wait(0.6)

        formula = MathTex(
            r"x = (-1)^s \cdot 2^{e-\mathrm{bias}} \cdot (1.m)",
            color=FG,
        ).scale(0.96)
        formula_box = self.panel(8.8, 1.2, color=PANEL_2, stroke="#27425e")
        formula_group = VGroup(formula_box, formula)
        formula.move_to(formula_box.get_center())
        formula_group.next_to(hook_group, DOWN, buff=0.4)

        sign_label = self.txt("sign", font_size=20, color=FG, max_width=1.6)
        exp_label = self.txt("exponent = range", font_size=20, color=FG, max_width=2.9)
        man_label = self.txt("mantissa = precision", font_size=20, color=FG, max_width=3.2)
        labels = VGroup(sign_label, exp_label, man_label).arrange(RIGHT, buff=0.35)
        label_panels = VGroup(
            *[self.panel(obj.width + 0.35, 0.46, color="#16263a", stroke="#22374d") for obj in labels]
        )
        label_groups = VGroup(*[VGroup(panel, obj) for panel, obj in zip(label_panels, labels)])
        for grp in label_groups:
            grp[1].move_to(grp[0].get_center())
        label_groups.arrange(RIGHT, buff=0.24)
        label_groups.next_to(formula_group, DOWN, buff=0.28)

        self.play(Write(formula), FadeIn(formula_box))
        self.play(LaggedStart(*(FadeIn(grp, shift=UP * 0.1) for grp in label_groups), lag_ratio=0.15))
        self.wait(0.7)

        self.play(FadeOut(hook_group, shift=UP * 0.12))

        fp16_card = self.format_card(
            "FP16",
            "16 bits   1 | 5 | 10",
            ["binary16", "more mantissa precision", "smaller exponent range"],
            FP16_C,
        )
        bf16_card = self.format_card(
            "BF16",
            "16 bits   1 | 8 | 7",
            ["bfloat16", "FP32-like exponent range", "coarser mantissa"],
            BF16_C,
        )
        fp8_card = self.format_card(
            "FP8 E4M3",
            "8 bits   1 | 4 | 3",
            ["smaller payload", "less precision", "typically paired with scaling"],
            FP8_C,
        )
        cards = VGroup(fp16_card, bf16_card, fp8_card).arrange(RIGHT, buff=0.28)
        cards.scale(0.9)
        cards.next_to(title, DOWN, buff=0.52)

        self.play(
            FadeOut(VGroup(formula_group, label_groups), shift=UP * 0.15),
            LaggedStart(*(FadeIn(card, shift=UP * 0.15) for card in cards), lag_ratio=0.15),
        )
        self.wait(0.4)

        fp16_bar = self.bit_bar("FP16", [1, 5, 10], [SIGN_C, FP16_C, BLUE_E], width=3.9)
        bf16_bar = self.bit_bar("BF16", [1, 8, 7], [SIGN_C, BF16_C, GREEN_E], width=3.9)
        fp8_bar = self.bit_bar("FP8 E4M3", [1, 4, 3], [SIGN_C, FP8_C, GOLD_E], width=3.9)
        bars = VGroup(fp16_bar, bf16_bar, fp8_bar).arrange(RIGHT, buff=0.35).scale(0.88)
        bars.next_to(cards, DOWN, buff=0.38)
        self.play(FadeIn(bars, shift=UP * 0.15))

        compare_note = self.txt(
            "BF16 keeps exponent range. FP16 keeps more mantissa precision.",
            font_size=22,
            color=MUTED,
            max_width=8.6,
        )
        note_panel = self.panel(compare_note.width + 0.45, 0.55, color=PANEL, stroke="#20364b")
        compare_group = VGroup(note_panel, compare_note)
        compare_note.move_to(note_panel.get_center())
        compare_group.next_to(bars, DOWN, buff=0.22)
        self.play(FadeIn(compare_group, shift=UP * 0.1))
        self.wait(0.8)

        mxfp8_title = self.txt("MXFP8: payload plus block scaling", font_size=27, color=FG, weight=BOLD, max_width=4.8)
        mxfp8_panel = self.panel(5.4, 2.05, color=PANEL_2, stroke=MXFP8_C)
        mxfp8_title.move_to(mxfp8_panel.get_top() + DOWN * 0.3)
        mxfp8_sub = self.txt("FP8 values + one E8M0 scale per 32 values", font_size=18, color=FG, max_width=4.8)
        mxfp8_sub.next_to(mxfp8_title, DOWN, buff=0.14)
        mxfp8_card2 = VGroup(mxfp8_panel, mxfp8_title, mxfp8_sub)

        block32 = self.block_grid(32, 16, MXFP8_C, side=0.23)
        block32.next_to(mxfp8_card2, DOWN, buff=0.3)
        block32_caption = self.txt("32 values", font_size=18, color=MUTED).next_to(block32, UP, buff=0.12)
        scale32_panel = self.panel(2.35, 0.46, color="#201a31", stroke=MXFP8_C)
        scale32_text = self.txt("1 shared E8M0 scale", font_size=17, color=FG, max_width=2.1)
        scale32_text.move_to(scale32_panel.get_center())
        scale32 = VGroup(scale32_panel, scale32_text).next_to(block32, DOWN, buff=0.14)
        mxfp8_group = VGroup(mxfp8_card2, block32_caption, block32, scale32)

        e5m2_note = self.txt("FP8 E5M2 exists too: more range, less mantissa.", font_size=18, color=MUTED, max_width=5.2)
        e5m2_note.next_to(mxfp8_group, DOWN, buff=0.18)

        self.play(
            FadeOut(VGroup(cards, bars, compare_group), shift=UP * 0.15),
            FadeIn(mxfp8_card2, shift=UP * 0.15),
            FadeIn(block32_caption, shift=UP * 0.1),
            FadeIn(e5m2_note, shift=UP * 0.1),
        )
        self.play(LaggedStartMap(FadeIn, block32, lag_ratio=0.04))
        self.play(FadeIn(scale32, shift=UP * 0.1))
        self.wait(0.8)

        nv_card = self.panel(5.8, 2.15, color=PANEL_2, stroke=NVFP4_C)
        nv_title = self.txt("NVFP4: 4-bit payload + two scales", font_size=26, color=FG, weight=BOLD, max_width=5.2)
        nv_title.move_to(nv_card.get_top() + DOWN * 0.32)
        nv_sub = self.txt("E2M1 payload + FP8 block scale + FP32 tensor scale", font_size=17, color=FG, max_width=5.2)
        nv_sub.next_to(nv_title, DOWN, buff=0.14)
        nv_header = VGroup(nv_card, nv_title, nv_sub)

        block16 = self.block_grid(16, 8, NVFP4_C, side=0.28)
        block16.next_to(nv_header, DOWN, buff=0.3)
        block16_caption = self.txt("16 payload values", font_size=18, color=MUTED).next_to(block16, UP, buff=0.12)
        block_scale_panel = self.panel(2.8, 0.46, color="#311c22", stroke=NVFP4_C)
        block_scale_text = self.txt("1 FP8 E4M3 block scale", font_size=16, color=FG, max_width=2.55).move_to(block_scale_panel.get_center())
        block_scale = VGroup(block_scale_panel, block_scale_text).next_to(block16, DOWN, buff=0.14)
        global_panel = self.panel(2.6, 0.46, color="#2a1620", stroke="#ff9f80")
        global_text = self.txt("1 FP32 tensor scale", font_size=16, color=FG, max_width=2.35).move_to(global_panel.get_center())
        global_scale = VGroup(global_panel, global_text).next_to(block_scale, DOWN, buff=0.12)
        nv_formula = MathTex(
            r"x = x_{\mathrm{E2M1}} \cdot s_{\mathrm{block}} \cdot s_{\mathrm{global}}",
            color=FG,
        ).scale(0.88)
        nv_formula.next_to(global_scale, DOWN, buff=0.22)
        nv_group = VGroup(nv_header, block16_caption, block16, block_scale, global_scale, nv_formula)

        self.play(
            FadeOut(VGroup(mxfp8_group, e5m2_note), shift=UP * 0.15),
            FadeIn(nv_header, shift=UP * 0.15),
            FadeIn(block16_caption, shift=UP * 0.1),
        )
        self.play(LaggedStartMap(FadeIn, block16, lag_ratio=0.05))
        self.play(
            FadeIn(block_scale, shift=UP * 0.08),
            FadeIn(global_scale, shift=UP * 0.08),
        )
        self.play(Write(nv_formula))
        self.wait(0.9)

        size_title = self.txt("Raw weight footprint by format", font_size=30, color=FG, weight=BOLD, max_width=8.2)
        size_note = self.txt(
            "Rough payload estimate only. Real deployment size also depends on scales and layout.",
            font_size=18,
            color=MUTED,
            max_width=8.8,
        )
        size_head = VGroup(size_title, size_note).arrange(DOWN, buff=0.12)
        size_head.next_to(title, DOWN, buff=0.5)

        row7 = self.model_row("7B model", [14.0, 7.0, 3.5], ["~14 GB", "~7 GB", "~3.5 GB"], 14.0)
        row70 = self.model_row("70B model", [140.0, 70.0, 35.0], ["~140 GB", "~70 GB", "~35 GB"], 140.0)
        row671 = self.model_row("671B class", [1340.0, 671.0, 335.5], ["~1.34 TB", "~671 GB", "~335.5 GB"], 1340.0)
        rows = VGroup(row7, row70, row671).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rows.scale(0.93)
        rows.next_to(size_head, DOWN, buff=0.38)

        legend_items = VGroup(
            *[
                VGroup(
                    Square(side_length=0.18, stroke_width=0, fill_color=color, fill_opacity=1),
                    self.txt(label, font_size=18, color=FG, max_width=1.8),
                ).arrange(RIGHT, buff=0.1)
                for color, label in [(FP16_C, "FP16/BF16"), (FP8_C, "FP8"), (NVFP4_C, "NVFP4 raw payload")]
            ]
        ).arrange(RIGHT, buff=0.3)
        legend_panel = self.panel(legend_items.width + 0.35, 0.48, color=PANEL, stroke="#24384f")
        legend = VGroup(legend_panel, legend_items)
        legend_items.move_to(legend_panel.get_center())
        legend.next_to(rows, DOWN, buff=0.24)

        self.play(
            FadeOut(nv_group, shift=UP * 0.15),
            FadeIn(size_head, shift=UP * 0.12),
            LaggedStart(FadeIn(row7, shift=UP * 0.1), FadeIn(row70, shift=UP * 0.1), FadeIn(row671, shift=UP * 0.1), lag_ratio=0.18),
            FadeIn(legend, shift=UP * 0.1),
        )
        self.wait(1.0)

        summary = Table(
            [
                ["FP16", "16", "more precision", "none", "~2 bytes/param"],
                ["BF16", "16", "larger range", "none", "~2 bytes/param"],
                ["FP8 E4M3", "8", "smaller payload", "usually yes", "~1 byte/param"],
                ["MXFP8", "8 + scales", "better local fit", "block scale", "slightly > FP8 raw"],
                ["NVFP4", "4 + scales", "very low payload", "block + tensor scale", "slightly > FP4 raw"],
            ],
            col_labels=[
                self.txt("Format", font_size=20, color=FG),
                self.txt("Bits", font_size=20, color=FG),
                self.txt("Profile", font_size=20, color=FG),
                self.txt("Scaling", font_size=20, color=FG),
                self.txt("Footprint", font_size=20, color=FG),
            ],
            include_outer_lines=True,
            line_config={"stroke_color": "#28425c", "stroke_width": 1},
            element_to_mobject=lambda text: self.txt(text, color=FG, font_size=17, max_width=2.7),
        ).scale(0.54)
        summary.next_to(title, DOWN, buff=0.55)

        outro = self.txt(
            "More bits buy precision. Better scaling buys usable low precision.",
            font_size=24,
            color=MUTED,
            weight=BOLD,
            max_width=9.2,
        )
        outro_panel = self.panel(outro.width + 0.45, 0.62, color=PANEL, stroke="#21384d")
        outro_group = VGroup(outro_panel, outro)
        outro.move_to(outro_panel.get_center())
        outro_group.next_to(summary, DOWN, buff=0.22)

        self.play(
            FadeOut(VGroup(size_head, rows, legend), shift=UP * 0.15),
            FadeIn(summary, shift=UP * 0.15),
            FadeIn(outro_group, shift=UP * 0.12),
        )
        self.wait(1.3)
