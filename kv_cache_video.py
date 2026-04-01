from manim import *


BG = "#08111D"
PANEL = "#102033"
PANEL_2 = "#13263D"
PANEL_3 = "#0C1828"
FG = "#F5F7FB"
MUTED = "#A8B4C7"
GRID = "#24384F"
Q_C = "#F5B041"
K_C = "#5DADE2"
V_C = "#58D68D"
WARN = "#EC7063"
GOOD = "#5AD1A4"
TOKEN_FILL = "#1A2B42"
FONT = "Avenir Next"


class KVCacheExplainer(Scene):
    def setup(self):
        self.camera.background_color = BG

    def txt(self, content, font_size=24, color=FG, weight=NORMAL, max_width=None):
        text = Text(content, font=FONT, font_size=font_size, color=color, weight=weight)
        if max_width and text.width > max_width:
            text.scale_to_fit_width(max_width)
        return text

    def panel(self, width, height, color=PANEL, stroke=GRID, opacity=0.97):
        return RoundedRectangle(
            corner_radius=0.18,
            width=width,
            height=height,
            fill_color=color,
            fill_opacity=opacity,
            stroke_color=stroke,
            stroke_width=1.5,
        )

    def token_box(self, label, width=1.2, height=0.62, fill=TOKEN_FILL, stroke="#35506E", text_color=FG):
        box = RoundedRectangle(
            corner_radius=0.12,
            width=width,
            height=height,
            fill_color=fill,
            fill_opacity=1,
            stroke_color=stroke,
            stroke_width=1.6,
        )
        text = self.txt(label, font_size=22, color=text_color, weight=MEDIUM, max_width=width - 0.18)
        text.move_to(box.get_center())
        return VGroup(box, text)

    def qkv_triplet(self, current=False):
        q = self.small_cell("Q", Q_C, active=current)
        k = self.small_cell("K", K_C, active=True)
        v = self.small_cell("V", V_C, active=True)
        triplet = VGroup(q, k, v).arrange(DOWN, buff=0.08)
        return triplet

    def small_cell(self, label, color, active=True):
        rect = RoundedRectangle(
            corner_radius=0.08,
            width=0.62,
            height=0.34,
            fill_color=color,
            fill_opacity=0.28 if active else 0.12,
            stroke_color=color,
            stroke_width=1.3 if active else 0.8,
        )
        text = self.txt(label, font_size=18, color=FG, weight=BOLD)
        text.move_to(rect.get_center())
        return VGroup(rect, text)

    def kv_pair(self, dimmed=False):
        opacity = 0.16 if dimmed else 0.26
        stroke_w = 1.0 if dimmed else 1.3
        k = RoundedRectangle(
            corner_radius=0.06,
            width=0.7,
            height=0.28,
            fill_color=K_C,
            fill_opacity=opacity,
            stroke_color=K_C,
            stroke_width=stroke_w,
        )
        v = RoundedRectangle(
            corner_radius=0.06,
            width=0.7,
            height=0.28,
            fill_color=V_C,
            fill_opacity=opacity,
            stroke_color=V_C,
            stroke_width=stroke_w,
        )
        k_label = self.txt("K", font_size=15, weight=BOLD).move_to(k.get_center())
        v_label = self.txt("V", font_size=15, weight=BOLD).move_to(v.get_center())
        return VGroup(VGroup(k, k_label), VGroup(v, v_label)).arrange(DOWN, buff=0.05)

    def work_meter(self, title, color, fill_ratio, width=3.8):
        label = self.txt(title, font_size=20, color=FG, weight=BOLD, max_width=width)
        track = RoundedRectangle(
            corner_radius=0.08,
            width=width,
            height=0.26,
            fill_color="#17283E",
            fill_opacity=1,
            stroke_width=0,
        )
        fill = RoundedRectangle(
            corner_radius=0.08,
            width=max(0.18, width * fill_ratio),
            height=0.26,
            fill_color=color,
            fill_opacity=1,
            stroke_width=0,
        )
        fill.align_to(track, LEFT)
        meter = VGroup(label, VGroup(track, fill)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        return meter

    def build_token_row(self, labels, highlight_last=False, empty_last=False):
        tokens = VGroup()
        for i, label in enumerate(labels):
            fill = TOKEN_FILL
            stroke = "#35506E"
            if empty_last and i == len(labels) - 1:
                fill = "#122033"
                stroke = Q_C
            elif highlight_last and i == len(labels) - 1:
                fill = "#2A2415"
                stroke = Q_C
            tokens.add(self.token_box(label, fill=fill, stroke=stroke))
        tokens.arrange(RIGHT, buff=0.18)
        return tokens

    def build_layer_view(self, labels):
        tokens = self.build_token_row(labels, highlight_last=True)
        token_to_triplet = VGroup()
        for i, token in enumerate(tokens):
            triplet = self.qkv_triplet(current=(i == len(tokens) - 1))
            triplet.next_to(token, DOWN, buff=0.28)
            token_to_triplet.add(triplet)
        guides = VGroup(
            *[
                Line(
                    token.get_bottom(),
                    triplet.get_top(),
                    color="#36506C",
                    stroke_width=1.2,
                )
                for token, triplet in zip(tokens, token_to_triplet)
            ]
        )
        layer_label = self.txt("One decoder layer", font_size=22, color=MUTED, max_width=3.4)
        layer_label.next_to(tokens, UP, buff=0.18).align_to(tokens, LEFT)
        layer = VGroup(layer_label, tokens, guides, token_to_triplet)
        return layer

    def make_step_card(self, step_label, token_count, recompute=True, labels=None, width=3.2, height=2.6):
        card = self.panel(width, height, color=PANEL_2, stroke="#29405A")
        title = self.txt(step_label, font_size=20, color=FG, weight=BOLD, max_width=width - 0.35)
        title.move_to(card.get_top() + DOWN * 0.28)
        if labels is None:
            labels = ["t"] * token_count
        token_row = self.build_token_row(labels, highlight_last=True)
        token_row.scale_to_fit_width(width - 0.35)
        token_row.next_to(title, DOWN, buff=0.18)

        kv_cols = VGroup()
        for _ in range(token_count):
            kv_cols.add(self.kv_pair(dimmed=not recompute))
        kv_cols.arrange(RIGHT, buff=0.08)
        kv_cols.scale_to_fit_width(width - 0.45)
        kv_cols.next_to(token_row, DOWN, buff=0.2)

        recompute_box = self.panel(width - 0.5, 0.4, color="#2B1A1E" if recompute else "#163026", stroke=WARN if recompute else GOOD)
        recompute_text = self.txt(
            "recompute whole prefix" if recompute else "reuse cached prefix",
            font_size=16,
            color=FG,
            max_width=width - 0.8,
        )
        recompute_text.move_to(recompute_box.get_center())
        recompute_group = VGroup(recompute_box, recompute_text)
        recompute_group.next_to(kv_cols, DOWN, buff=0.18)

        return VGroup(card, title, token_row, kv_cols, recompute_group)

    def cache_bank(self, num_layers=3, tokens_per_row=6):
        rows = VGroup()
        for layer_idx in range(num_layers):
            layer_label = self.txt(f"Layer {layer_idx + 1}", font_size=18, color=MUTED, max_width=1.1)
            cells = VGroup(*[self.kv_pair(dimmed=False).scale(0.56) for _ in range(tokens_per_row)])
            cells.arrange(RIGHT, buff=0.07)
            row = VGroup(layer_label, cells).arrange(RIGHT, buff=0.18, aligned_edge=UP)
            rows.add(row)
        rows.arrange(DOWN, buff=0.18, aligned_edge=LEFT)
        box = self.panel(rows.width + 0.5, rows.height + 0.82, color=PANEL_3, stroke="#37506E")
        title = self.txt("KV cache", font_size=25, color=FG, weight=BOLD, max_width=box.width - 0.4)
        title.move_to(box.get_top() + DOWN * 0.28)
        rows.next_to(title, DOWN, buff=0.24)
        return VGroup(box, title, rows)

    def construct(self):
        title_panel = self.panel(7.3, 1.02, color="#0B1728", stroke="#22384F")
        title = self.txt("Why KV Cache Matters", font_size=34, color=FG, weight=BOLD, max_width=6.7)
        subtitle = self.txt("recompute vs reuse during decode", font_size=20, color=MUTED, max_width=6.6)
        title.move_to(title_panel.get_center() + UP * 0.14)
        subtitle.move_to(title_panel.get_center() + DOWN * 0.2)
        title_group = VGroup(title_panel, title, subtitle).to_edge(UP, buff=0.28)

        token_row = self.build_token_row(["The", "model", "writes", "next", "?"], empty_last=True)
        token_row.next_to(title_group, DOWN, buff=0.55)
        note = self.txt("next token depends on full prefix", font_size=24, color=MUTED, max_width=7.5)
        note.next_to(token_row, DOWN, buff=0.28)

        arcs = VGroup()
        target = token_row[-1]
        for token in token_row[:-1]:
            arc = CubicBezier(
                target.get_top() + LEFT * 0.18,
                target.get_top() + UP * 0.95,
                token.get_top() + UP * 0.95,
                token.get_top() + RIGHT * 0.18,
                color=Q_C,
                stroke_opacity=0.35,
                stroke_width=2.0,
            )
            arcs.add(arc)

        self.play(FadeIn(title_group, shift=UP * 0.15))
        self.play(LaggedStart(*(FadeIn(tok, shift=UP * 0.1) for tok in token_row), lag_ratio=0.12))
        self.play(LaggedStart(*(Create(arc) for arc in arcs), lag_ratio=0.1), FadeIn(note, shift=UP * 0.08), run_time=1.8)
        self.wait(0.4)

        layer = self.build_layer_view(["The", "model", "writes", "next", "token"])
        layer.move_to(DOWN * 0.15)
        layer_tokens = layer[1]
        triplets = layer[3]
        q_box = triplets[-1][0]
        ks = VGroup(*[triplet[1] for triplet in triplets[:-1]])
        vs = VGroup(*[triplet[2] for triplet in triplets[:-1]])
        q_to_k = VGroup(
            *[
                Arrow(
                    q_box.get_left(),
                    k.get_right(),
                    buff=0.06,
                    color=Q_C,
                    stroke_width=1.8,
                    max_tip_length_to_length_ratio=0.08,
                )
                for k in ks
            ]
        )
        v_pull = VGroup(
            *[
                Arrow(
                    v.get_right(),
                    q_box.get_bottom() + DOWN * 0.18,
                    buff=0.06,
                    color=V_C,
                    stroke_width=1.5,
                    max_tip_length_to_length_ratio=0.08,
                )
                for v in vs
            ]
        )

        self.play(
            FadeOut(arcs, shift=UP * 0.05),
            FadeOut(note, shift=UP * 0.05),
            Transform(token_row, layer_tokens),
            FadeIn(layer[0], shift=UP * 0.08),
            FadeIn(layer[2], shift=UP * 0.08),
            LaggedStart(*(FadeIn(triplet, shift=UP * 0.08) for triplet in triplets), lag_ratio=0.08),
            run_time=1.7,
        )
        self.play(LaggedStart(*(Create(a) for a in q_to_k), lag_ratio=0.08), run_time=1.0)
        self.play(LaggedStart(*(Create(a) for a in v_pull), lag_ratio=0.08), run_time=1.0)
        self.wait(0.3)

        self.play(FadeOut(q_to_k), FadeOut(v_pull))

        prefix_strip_panel = self.panel(8.9, 0.86, color="#0B1728", stroke="#20354C")
        prefix_strip = self.build_token_row(["The", "model", "writes", "next", "token", "..."], highlight_last=True)
        prefix_strip.scale_to_fit_width(8.1)
        prefix_strip.move_to(prefix_strip_panel.get_center())
        prefix_note = self.txt("same prefix keeps getting rebuilt", font_size=18, color=MUTED, max_width=5.2)
        prefix_note.next_to(prefix_strip_panel, DOWN, buff=0.12)
        prefix_group = VGroup(prefix_strip_panel, prefix_strip, prefix_note)
        prefix_group.next_to(title_group, DOWN, buff=0.52)

        no_cache_title = self.txt("Without KV cache", font_size=30, color=WARN, weight=BOLD, max_width=4.2)
        no_cache_title.next_to(prefix_group, DOWN, buff=0.3).align_to(prefix_group, LEFT)
        step1 = self.make_step_card("decode step 1", 4, recompute=True, labels=["The", "model", "writes", "next"], width=2.8, height=2.42)
        step2 = self.make_step_card("decode step 2", 5, recompute=True, labels=["The", "model", "writes", "next", "token"], width=2.8, height=2.42)
        step3 = self.make_step_card("decode step 3", 6, recompute=True, labels=["The", "model", "writes", "next", "token", "..."], width=2.8, height=2.42)
        no_cache_cards = VGroup(step1, step2, step3).arrange(RIGHT, buff=0.18, aligned_edge=UP)
        no_cache_cards.scale(0.74)
        no_cache_cards.next_to(no_cache_title, DOWN, buff=0.24)
        no_cache_cards.align_to(prefix_group, LEFT).shift(RIGHT * 0.12)
        heavy_meter = self.work_meter("work grows fast", WARN, 0.9)
        heavy_meter.scale(0.8)
        heavy_meter.next_to(no_cache_cards, DOWN, buff=0.18)
        heavy_meter.align_to(no_cache_cards, LEFT)

        transforms = []
        for src, dst in zip(layer_tokens, step1[2]):
            transforms.append(TransformFromCopy(src, dst))

        self.play(
            FadeOut(layer[0], shift=UP * 0.08),
            FadeOut(layer[2], shift=UP * 0.08),
            FadeOut(triplets, shift=UP * 0.08),
            FadeOut(token_row, shift=UP * 0.06),
            FadeIn(prefix_group, shift=UP * 0.08),
            FadeIn(no_cache_title, shift=UP * 0.1),
            LaggedStart(*(FadeIn(card, shift=UP * 0.08) for card in [step1[0], step2[0], step3[0]]), lag_ratio=0.08),
            LaggedStart(*(FadeIn(title, shift=UP * 0.08) for title in [step1[1], step2[1], step3[1]]), lag_ratio=0.08),
            LaggedStart(*(FadeIn(tag, shift=UP * 0.08) for tag in [step1[4], step2[4], step3[4]]), lag_ratio=0.08),
            *transforms,
            run_time=1.4,
        )
        self.play(
            LaggedStart(*(FadeIn(kv, shift=UP * 0.04) for kv in step1[3]), lag_ratio=0.08),
            LaggedStart(*(FadeIn(tok, shift=UP * 0.04) for tok in step2[2]), lag_ratio=0.04),
            LaggedStart(*(FadeIn(tok, shift=UP * 0.04) for tok in step3[2]), lag_ratio=0.04),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*(FadeIn(kv, shift=UP * 0.04) for kv in step2[3]), lag_ratio=0.06),
            LaggedStart(*(FadeIn(kv, shift=UP * 0.04) for kv in step3[3]), lag_ratio=0.06),
            FadeIn(heavy_meter, shift=UP * 0.08),
            run_time=1.0,
        )
        flash_anims = [Indicate(kv, color=WARN, scale_factor=1.05) for kv in step1[3][:-1]]
        flash_anims_2 = [Indicate(kv, color=WARN, scale_factor=1.05) for kv in step2[3][:-1]]
        flash_anims_3 = [Indicate(kv, color=WARN, scale_factor=1.05) for kv in step3[3][:-1]]
        self.play(
            LaggedStart(*flash_anims, lag_ratio=0.08),
            LaggedStart(*flash_anims_2, lag_ratio=0.06),
            LaggedStart(*flash_anims_3, lag_ratio=0.05),
            run_time=1.3,
        )
        self.wait(0.3)

        cache_title = self.txt("With KV cache", font_size=30, color=GOOD, weight=BOLD, max_width=4.2)
        cache_title.next_to(title_group, DOWN, buff=0.5).align_to(title_group, LEFT)
        cache_bank = self.cache_bank(num_layers=3, tokens_per_row=6)
        cache_bank.scale(0.82)
        cache_bank.to_edge(RIGHT, buff=0.42).shift(DOWN * 0.28)

        cached_step1 = self.make_step_card("decode step 1", 4, recompute=False, labels=["The", "model", "writes", "next"], width=2.9, height=2.32)
        cached_step2 = self.make_step_card("decode step 2", 5, recompute=False, labels=["The", "model", "writes", "next", "token"], width=2.9, height=2.32)
        cached_step3 = self.make_step_card("decode step 3", 6, recompute=False, labels=["The", "model", "writes", "next", "token", "..."], width=2.9, height=2.32)
        cached_steps = VGroup(cached_step1, cached_step2, cached_step3).arrange(DOWN, buff=0.18)
        cached_steps.scale(0.72)
        cached_steps.to_edge(LEFT, buff=0.48).shift(DOWN * 0.22)
        light_meter = self.work_meter("only append new state", GOOD, 0.34, width=3.2)
        light_meter.scale(0.82)
        light_meter.next_to(cached_steps, DOWN, buff=0.26)

        kv_sources = VGroup(*step3[3][:4])
        cache_targets = VGroup(*cache_bank[2][0][1][:4])
        cache_copies = [src.copy() for src in kv_sources]
        self.add(*cache_copies)

        self.play(
            FadeOut(prefix_group, shift=UP * 0.08),
            FadeOut(no_cache_title, shift=UP * 0.08),
            FadeOut(heavy_meter, shift=DOWN * 0.08),
            FadeOut(step2, shift=LEFT * 0.1),
            FadeOut(step3, shift=LEFT * 0.1),
            ReplacementTransform(step1, cached_step1),
            FadeIn(cached_step2, shift=LEFT * 0.1),
            FadeIn(cached_step3, shift=LEFT * 0.1),
            FadeIn(cache_title, shift=UP * 0.08),
            FadeIn(cache_bank, shift=RIGHT * 0.15),
            FadeIn(light_meter, shift=UP * 0.08),
            run_time=1.5,
        )
        self.play(
            LaggedStart(
                *[Transform(copy, target.copy()) for copy, target in zip(cache_copies, cache_targets)],
                lag_ratio=0.08,
            ),
            run_time=1.0,
        )
        self.remove(*cache_copies)

        current_q = cached_step3[3][-1][0]
        read_lines = VGroup(
            *[
                DashedLine(
                    current_q.get_right(),
                    cell.get_left(),
                    dash_length=0.12,
                    color=Q_C,
                    stroke_width=1.4,
                )
                for cell in cache_bank[2][0][1][:4]
            ]
        )
        new_append_targets = VGroup(
            cache_bank[2][0][1][4],
            cache_bank[2][1][1][4],
            cache_bank[2][2][1][4],
        )
        append_sources = VGroup(*[self.kv_pair(dimmed=False).scale(0.5) for _ in range(3)])
        for src, step in zip(append_sources, cached_steps):
            src.next_to(step[3][-1], RIGHT, buff=0.12)

        self.play(LaggedStart(*(Create(line) for line in read_lines), lag_ratio=0.08), run_time=0.9)
        self.play(LaggedStart(*(FadeIn(src, shift=LEFT * 0.08) for src in append_sources), lag_ratio=0.1), run_time=0.8)
        self.play(
            LaggedStart(
                *[Transform(src, target.copy()) for src, target in zip(append_sources, new_append_targets)],
                lag_ratio=0.08,
            ),
            run_time=0.9,
        )
        self.play(FadeOut(read_lines), run_time=0.4)
        self.wait(0.3)

        left_box = self.panel(5.6, 3.45, color=PANEL_2, stroke=WARN)
        right_box = self.panel(5.6, 3.45, color=PANEL_2, stroke=GOOD)
        left_box.to_edge(LEFT, buff=0.3).shift(DOWN * 0.15)
        right_box.to_edge(RIGHT, buff=0.3).shift(DOWN * 0.15)
        left_label = self.txt("Recompute full prefix", font_size=26, color=WARN, weight=BOLD, max_width=5.2)
        right_label = self.txt("Reuse cached K/V", font_size=26, color=GOOD, weight=BOLD, max_width=5.2)
        left_label.move_to(left_box.get_top() + DOWN * 0.3)
        right_label.move_to(right_box.get_top() + DOWN * 0.3)

        left_tokens = self.build_token_row(["The", "model", "writes", "next", "token", "..."], highlight_last=True)
        left_tokens.scale(0.62).next_to(left_label, DOWN, buff=0.2)
        right_tokens = left_tokens.copy().next_to(right_label, DOWN, buff=0.2)

        left_kvs = VGroup(*[self.kv_pair(dimmed=False).scale(0.56) for _ in range(6)]).arrange(RIGHT, buff=0.05)
        right_kvs = VGroup(*[self.kv_pair(dimmed=False).scale(0.56) for _ in range(6)]).arrange(RIGHT, buff=0.05)
        left_kvs.next_to(left_tokens, DOWN, buff=0.28)
        right_kvs.next_to(right_tokens, DOWN, buff=0.28)

        left_meter = self.work_meter("recompute known work", WARN, 0.92, width=3.4).scale(0.9)
        right_meter = self.work_meter("append one step", GOOD, 0.28, width=3.4).scale(0.9)
        left_meter.next_to(left_kvs, DOWN, buff=0.28)
        right_meter.next_to(right_kvs, DOWN, buff=0.28)

        final_caption = self.txt(
            "KV cache keeps past attention state so decode stays tractable.",
            font_size=24,
            color=FG,
            weight=BOLD,
            max_width=10.8,
        )
        final_caption.to_edge(DOWN, buff=0.25)

        self.play(
            FadeOut(VGroup(cache_title, cached_steps, cache_bank, light_meter), shift=UP * 0.08),
            FadeIn(VGroup(left_box, right_box, left_label, right_label), shift=UP * 0.08),
            FadeIn(VGroup(left_tokens, right_tokens, left_kvs, right_kvs), shift=UP * 0.08),
            FadeIn(VGroup(left_meter, right_meter), shift=UP * 0.08),
            run_time=1.4,
        )
        self.play(
            LaggedStart(*[Indicate(kv, color=WARN, scale_factor=1.04) for kv in left_kvs[:-1]], lag_ratio=0.05),
            Indicate(right_kvs[-1], color=GOOD, scale_factor=1.04),
            run_time=1.1,
        )
        self.play(FadeIn(final_caption, shift=UP * 0.08))
        self.wait(1.2)
