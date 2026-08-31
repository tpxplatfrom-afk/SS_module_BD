# এইচএসসি (HSC) উচ্চতর গণিত — ১ম ও ২য় পত্র পূর্ণাঙ্গ হ্যান্ডবুক

## ১. ক্যালকুলাস — অন্তরীকরণ (Differentiation / Derivative)
- মূল নিয়মে অন্তরজের সংজ্ঞা: f'(x) = lim_{h -> 0} [ {f(x + h) - f(x)} / h ]
- গুরুত্বপূর্ণ মৌলিক অন্তরজ সূত্রসমূহ:
  * d/dx (x^n) = n * x^(n - 1)
  * d/dx (e^x) = e^x,  d/dx (a^x) = a^x * ln(a)
  * d/dx (ln x) = 1/x, d/dx (log_a x) = 1 / (x * ln a)
  * d/dx (sin x) = cos x,  d/dx (cos x) = -sin x
  * d/dx (tan x) = sec² x,  d/dx (cot x) = -cosec² x
  * d/dx (sec x) = sec x * tan x,  d/dx (cosec x) = -cosec x * cot x
  * d/dx (sin⁻¹ x) = 1 / √(1 - x²),  d/dx (tan⁻¹ x) = 1 / (1 + x²)
- গুণের সূত্র (Product Rule): d/dx (uv) = u (dv/dx) + v (du/dx)
- ভাগের সূত্র (Quotient Rule): d/dx (u/v) = [ v (du/dx) - u (dv/dx) ] / v²
- চেইন রুল (Chain Rule): dy/dx = (dy/du) * (du/dx)
- গুরুমান ও লঘুমান (Maxima & Minima): f'(x) = 0 হলে চরম মান পাওয়া যায়। f''(x) < 0 হলে গুরুমান (Maximum) এবং f''(x) > 0 হলে লঘুমান (Minimum)।

## ২. ক্যালকুলাস — যোগজীকরণ (Integration / Anti-Derivative)
- অনির্দিষ্ট যোগজের সূত্রসমূহ:
  * ∫ x^n dx = [ x^(n + 1) / (n + 1) ] + C  (যেখানে n != -1)
  * ∫ (1/x) dx = ln|x| + C
  * ∫ e^x dx = e^x + C,  ∫ a^x dx = (a^x / ln a) + C
  * ∫ sin x dx = -cos x + C,  ∫ cos x dx = sin x + C
  * ∫ sec² x dx = tan x + C,  ∫ cosec² x dx = -cot x + C
  * ∫ sec x dx = ln|sec x + tan x| + C
  * ∫ 1 / (a² + x²) dx = (1/a) * tan⁻¹(x/a) + C
  * ∫ 1 / √(a² - x²) dx = sin⁻¹(x/a) + C
- অংশায়ন সূত্র (Integration by Parts / LIATE Rule):
  * ∫ u v dx = u ∫ v dx - ∫ [ (du/dx) * ∫ v dx ] dx
  * অগ্রগণ্য ক্রম: L (Logarithmic) -> I (Inverse Trig) -> A (Algebraic) -> T (Trigonometric) -> E (Exponential)
- নির্দিষ্ট যোগজ ও ক্ষেত্রফল: Area = ∫_{a}^{b} y dx

## ৩. ম্যাট্রিক্স, নির্ণায়ক ও জটিল সংখ্যা
- ক্র্যামারের নিয়ম (Cramer's Rule): x = Dx/D, y = Dy/D, z = Dz/D
- বিপরীত ম্যাট্রিক্স: A⁻¹ = (1 / det(A)) * adj(A)
- জটিল সংখ্যা (Complex Numbers): z = x + iy, মডুলাস r = √(x² + y²), আর্গুমেন্ট θ = tan⁻¹(y/x), ডি ময়ভারের উপপাদ্য (cos θ + i sin θ)^n = cos(nθ) + i sin(nθ)।

## ৪. কণিক ও স্থিতিবিদ্যা-গতিবিদ্যা (Conics, Statics & Dynamics)
- পরাবৃত্ত: y² = 4ax (শীর্ষ (0,0), উপকেন্দ্র (a,0), নিয়ামক x = -a)
- উপবৃত্ত: x²/a² + y²/b² = 1 (উৎকেন্দ্রিকতা e = √(1 - b²/a²) < 1)
- অধিবৃত্ত: x²/a² - y²/b² = 1 (উৎকেন্দ্রিকতা e = √(1 + b²/a²) > 1)
- লামির উপপাদ্য (Lami's Theorem): P / sin α = Q / sin β = R / sin γ
- প্রাসের গতি (Projectile Motion): সর্বাধিক উচ্চতা H = (u² sin² α) / (2g), বিচরণকাল T = (2u sin α) / g, পাল্লা R = (u² sin 2α) / g।
