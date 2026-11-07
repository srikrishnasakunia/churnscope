"""ui_effects.py — motion and shader layer for the ChurnScope app.

Two pieces, both driven by the colours in .streamlit/config.toml:

1. shader_banner()  — a WebGL fragment-shader hero banner: domain-warped noise
   in the theme colours with drifting "signal" waves (a nod to telecom).
   Pauses when scrolled off-screen or when the tab is hidden, caps pixel
   density for battery, and draws one still frame for users who prefer
   reduced motion. Falls back to a CSS gradient if WebGL is unavailable.

2. inject_motion_css() — CSS animations for the rest of the page: content
   fades up as it appears, metric cards lift on hover, a light sheen sweeps
   across buttons, the slider thumb glows, and charts ease in. All of it is
   switched off under `prefers-reduced-motion`.
"""
import colorsys

import streamlit as st


# ---------- colour helpers ----------
def _rgb01(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _shift(hex_color: str, degrees: float, lightness: float | None = None) -> str:
    h, l, s = colorsys.rgb_to_hls(*_rgb01(hex_color))
    r, g, b = colorsys.hls_to_rgb((h + degrees / 360) % 1, l if lightness is None else lightness, s)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def _vec3(hex_color: str) -> str:
    return "vec3({:.4f}, {:.4f}, {:.4f})".format(*_rgb01(hex_color))


def _font_name(option: str, fallback: str) -> str:
    """'Playfair Display':https://… -> Playfair Display"""
    try:
        raw = st.get_option(option) or ""
    except Exception:
        raw = ""
    name = raw.split(":http")[0].strip().strip("'\"") if raw else ""
    return name or fallback


# ---------- 1. shader banner ----------
_FRAGMENT = """
precision mediump float;
uniform vec2  uRes;
uniform float uTime;
uniform vec3  uDeep;     // dark accent
uniform vec3  uAccent;   // theme accent
uniform vec3  uGlow;     // hue-shifted highlight

// value noise + fbm
float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash(i), hash(i+vec2(1,0)), u.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p){
  float v = 0.0, a = 0.5;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 5; i++){ v += a*noise(p); p = r*p*2.02; a *= 0.5; }
  return v;
}

void main(){
  vec2 uv = gl_FragCoord.xy / uRes.xy;
  vec2 p  = vec2(uv.x * uRes.x/uRes.y, uv.y) * 2.2;
  float t = uTime * 0.06;

  // domain warping: noise feeding noise gives the slow "ink in water" flow
  vec2 q = vec2(fbm(p + vec2(0.0, t)), fbm(p + vec2(5.2, 1.3) - t));
  vec2 r = vec2(fbm(p + 3.0*q + vec2(1.7, 9.2) + t*1.3), fbm(p + 3.0*q + vec2(8.3, 2.8) - t));
  float f = fbm(p + 3.0*r);

  vec3 col = mix(uDeep, uAccent, smoothstep(0.15, 0.85, f));
  col = mix(col, uGlow, smoothstep(0.55, 1.0, length(q)) * 0.55);

  // drifting signal waves — thin glowing sine lines
  for (int k = 0; k < 3; k++){
    float fk   = float(k);
    float y    = 0.5 + 0.18*sin(uv.x*(5.0+fk*2.3) + uTime*(0.5+fk*0.17) + fk*2.1)
                     + 0.05*sin(uv.x*17.0 - uTime*0.9 + fk);
    float line = 0.0025 / abs(uv.y - (y - 0.22 + fk*0.22));
    col += uGlow * clamp(line, 0.0, 1.0) * 0.35;
  }

  // left-side vignette keeps the headline readable
  col *= mix(0.55, 1.0, smoothstep(0.0, 0.75, uv.x));
  // fine film grain so gradients don't band
  col += (hash(gl_FragCoord.xy + uTime) - 0.5) * 0.025;

  gl_FragColor = vec4(col, 1.0);
}
"""

_VERTEX = "attribute vec2 a; void main(){ gl_Position = vec4(a, 0.0, 1.0); }"

_BANNER_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family={font_q}:wght@600;700&family={body_q}:wght@400;500&display=swap" rel="stylesheet">
<style>
  html, body {{ margin:0; padding:0; background:transparent; }}
  .hero {{
    position:relative; height:{height}px; border-radius:{radius}; overflow:hidden;
    background: linear-gradient(120deg, {deep}, {accent} 55%, {glow});   /* no-WebGL fallback */
    font-family:'{body}', system-ui, sans-serif; color:#fff;
    animation: heroIn .9s cubic-bezier(.2,.7,.2,1) both;
  }}
  canvas {{ position:absolute; inset:0; width:100%; height:100%; display:block; }}
  .copy {{ position:absolute; left:36px; bottom:30px; right:36px; }}
  .kicker {{ font-size:12px; letter-spacing:.18em; text-transform:uppercase; opacity:.85;
            animation: rise .8s .15s both; }}
  h1 {{ font-family:'{font}', Georgia, serif; font-weight:700; font-size:clamp(28px, 4.2vw, 46px);
       line-height:1.08; margin:6px 0 8px; text-shadow:0 2px 18px rgba(0,0,0,.25);
       animation: rise .8s .28s both; }}
  .sub {{ font-size:15px; opacity:.92; max-width:620px; animation: rise .8s .42s both; }}
  .chip {{ display:inline-block; margin-top:12px; padding:5px 12px; border-radius:999px;
          background:rgba(255,255,255,.16); backdrop-filter:blur(6px); font-size:12.5px;
          border:1px solid rgba(255,255,255,.28); animation: rise .8s .56s both; }}
  .dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:#fff;
         margin-right:7px; vertical-align:1px; animation: pulse 1.8s ease-in-out infinite; }}
  @keyframes heroIn {{ from {{ opacity:0; transform:scale(.985); }} to {{ opacity:1; transform:none; }} }}
  @keyframes rise   {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:none; }} }}
  @keyframes pulse  {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:.35; transform:scale(.7); }} }}
  @media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; }} }}
</style>
<div class="hero" role="img" aria-label="Animated banner: Will this customer leave?">
  <canvas id="fx"></canvas>
  <div class="copy">
    <div class="kicker">ChurnScope · Telco customer churn</div>
    <h1>Will this customer leave?</h1>
    <div class="sub">{subtitle}</div>
    <div class="chip"><span class="dot"></span>{chip}</div>
  </div>
</div>
<script>
(function () {{
  const canvas = document.getElementById('fx');
  const gl = canvas.getContext('webgl', {{ antialias:false, premultipliedAlpha:false }});
  if (!gl) {{ canvas.remove(); return; }}            // CSS gradient fallback stays visible

  function compile(type, src) {{
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {{ console.warn(gl.getShaderInfoLog(s)); return null; }}
    return s;
  }}
  const vs = compile(gl.VERTEX_SHADER, {vertex});
  const fs = compile(gl.FRAGMENT_SHADER, `{fragment}`);
  if (!vs || !fs) {{ canvas.remove(); return; }}
  const prog = gl.createProgram(); gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
  gl.useProgram(prog);

  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);  // one big triangle
  const loc = gl.getAttribLocation(prog, 'a');
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const uRes = gl.getUniformLocation(prog, 'uRes'), uTime = gl.getUniformLocation(prog, 'uTime');
  gl.uniform3fv(gl.getUniformLocation(prog, 'uDeep'),   [{deep_v}]);
  gl.uniform3fv(gl.getUniformLocation(prog, 'uAccent'), [{accent_v}]);
  gl.uniform3fv(gl.getUniformLocation(prog, 'uGlow'),   [{glow_v}]);

  function resize() {{
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);   // cap for battery
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h); }}
    gl.uniform2f(uRes, w, h);
  }}
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let visible = true, raf = 0, t0 = performance.now();
  function frame(now) {{
    resize();
    gl.uniform1f(uTime, still ? 12.0 : (now - t0) / 1000);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (!still && visible && !document.hidden) raf = requestAnimationFrame(frame);
  }}
  new IntersectionObserver(es => {{
    visible = es[0].isIntersecting;
    if (visible && !still) {{ cancelAnimationFrame(raf); raf = requestAnimationFrame(frame); }}
  }}).observe(canvas);
  document.addEventListener('visibilitychange', () => {{
    if (!document.hidden && visible && !still) {{ cancelAnimationFrame(raf); raf = requestAnimationFrame(frame); }}
  }});
  window.addEventListener('resize', () => requestAnimationFrame(frame));
  raf = requestAnimationFrame(frame);
}})();
</script>
"""


def shader_banner(accent: str, subtitle: str, chip: str, height: int = 230) -> None:
    deep = _shift(accent, -12, 0.16)
    glow = _shift(accent, 38, 0.62)
    heading = _font_name("theme.headingFont", "Georgia")
    body = _font_name("theme.font", "system-ui")
    try:
        radius = st.get_option("theme.baseRadius") or "0.6rem"
    except Exception:
        radius = "0.6rem"
    html = _BANNER_HTML.format(
        height=height, radius=radius, deep=deep, accent=accent, glow=glow,
        font=heading, font_q=heading.replace(" ", "+"), body=body, body_q=body.replace(" ", "+"),
        subtitle=subtitle, chip=chip,
        vertex=repr(_VERTEX), fragment=_FRAGMENT.strip(),
        deep_v="{:.4f},{:.4f},{:.4f}".format(*_rgb01(deep)),
        accent_v="{:.4f},{:.4f},{:.4f}".format(*_rgb01(accent)),
        glow_v="{:.4f},{:.4f},{:.4f}".format(*_rgb01(glow)),
    )
    # Newer Streamlit can run the script in-page; older versions use a sandboxed iframe.
    try:
        st.html(html, unsafe_allow_javascript=True)
    except (TypeError, AttributeError):  # param missing (<1.5x) or st.html missing (<1.36)
        import streamlit.components.v1 as components
        components.html(html, height=height + 8)


# ---------- 2. page-wide motion ----------
_CSS = """
<style>
:root {{ --accent: {accent}; --accent-soft: {accent_soft}; --cs-radius: {radius}; --ease: cubic-bezier(.2,.7,.2,1); }}

/* content fades up as it mounts (Streamlit keeps elements between reruns, so this plays once) */
[data-testid="stMain"] [data-testid="stElementContainer"] {{
  animation: cs-rise .55s var(--ease) both;
}}
[data-testid="stMain"] [data-testid="stColumn"]:nth-child(2) [data-testid="stElementContainer"] {{ animation-delay: .06s; }}
[data-testid="stMain"] [data-testid="stColumn"]:nth-child(3) [data-testid="stElementContainer"] {{ animation-delay: .12s; }}
[data-testid="stMain"] [data-testid="stColumn"]:nth-child(4) [data-testid="stElementContainer"] {{ animation-delay: .18s; }}
[data-testid="stMain"] [data-testid="stColumn"]:nth-child(5) [data-testid="stElementContainer"] {{ animation-delay: .24s; }}
[data-testid="stMain"] [data-testid="stColumn"]:nth-child(6) [data-testid="stElementContainer"] {{ animation-delay: .30s; }}

/* metric tiles become cards that lift on hover */
[data-testid="stMetric"] {{
  background: var(--accent-soft);
  border: 1px solid color-mix(in srgb, var(--accent) 18%, transparent);
  border-radius: var(--cs-radius, 0.6rem);
  padding: 12px 14px;
  transition: transform .25s var(--ease), box-shadow .25s var(--ease), border-color .25s;
}}
[data-testid="stMetric"]:hover {{
  transform: translateY(-3px);
  box-shadow: 0 10px 24px -12px color-mix(in srgb, var(--accent) 55%, transparent);
  border-color: color-mix(in srgb, var(--accent) 45%, transparent);
}}
[data-testid="stMetricValue"] {{ animation: cs-pop .5s var(--ease) both; }}

/* charts ease in */
[data-testid="stMain"] img {{ animation: cs-fade .7s var(--ease) both; }}

/* buttons: light sheen sweeps across on hover */
[data-testid="stDownloadButton"] button, [data-testid="stBaseButton-secondary"] {{
  position: relative; overflow: hidden;
  transition: transform .2s var(--ease), box-shadow .2s var(--ease);
}}
[data-testid="stDownloadButton"] button::after {{
  content: ""; position: absolute; top: 0; left: -60%; width: 40%; height: 100%;
  background: linear-gradient(100deg, transparent, rgba(255,255,255,.55), transparent);
  transform: skewX(-20deg); pointer-events: none;
}}
[data-testid="stDownloadButton"] button:hover {{ transform: translateY(-1px);
  box-shadow: 0 6px 16px -8px color-mix(in srgb, var(--accent) 60%, transparent); }}
[data-testid="stDownloadButton"] button:hover::after {{ animation: cs-sheen .8s var(--ease); }}

/* slider thumb breathes softly so it's easy to spot */
[data-testid="stSlider"] [role="slider"] {{ animation: cs-glow 2.4s ease-in-out infinite; }}

/* tabs: smooth colour change */
[data-testid="stTabs"] button[role="tab"] {{ transition: color .2s, background-color .2s; }}
[data-testid="stTabs"] button[role="tab"]:hover {{ background: var(--accent-soft); }}

/* expander & dataframe ease in */
[data-testid="stExpander"], [data-testid="stDataFrame"] {{ animation: cs-fade .6s var(--ease) both; }}

@keyframes cs-rise  {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes cs-fade  {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes cs-pop   {{ 0% {{ opacity: 0; transform: scale(.92); }} 60% {{ transform: scale(1.03); }} 100% {{ opacity: 1; transform: none; }} }}
@keyframes cs-sheen {{ from {{ left: -60%; }} to {{ left: 130%; }} }}
@keyframes cs-glow  {{
  0%, 100% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent) 45%, transparent); }}
  50%      {{ box-shadow: 0 0 0 7px color-mix(in srgb, var(--accent) 0%, transparent); }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def inject_motion_css(accent: str) -> None:
    soft = _shift(accent, 0, 0.965)
    try:
        radius = st.get_option("theme.baseRadius") or "0.6rem"
    except Exception:
        radius = "0.6rem"
    st.markdown(_CSS.format(accent=accent, accent_soft=soft, radius=radius), unsafe_allow_html=True)
