#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║   ASTRO-MÜHENDİSLİK VE FİZİKSEL MODELLEME AKADEMİSİ          ║
║   Ötegezegen Transit Analizi Atölye Programı                    ║
║                                                                  ║
║   Bölüm 1: Transit Fiziği ve Geometrik Optik                   ║
║   Bölüm 2: Limb Darkening ve Sinyal İşleme                     ║
║   Bölüm 3: Kepler Yasaları ve Fiziksel Karakterizasyon          ║
╚══════════════════════════════════════════════════════════════════╝

Gereksinimler:
    pip install numpy matplotlib scipy

Kullanım:
    python astro_muhendislik_atolye.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# GENEL AYARLAR
# ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#0d1117',
    'axes.edgecolor':   '#c9d1d9',
    'axes.labelcolor':  '#c9d1d9',
    'text.color':       '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'font.size':        11,
    'font.family':      'sans-serif',
})

# Fiziksel Sabitler
R_SUN_KM     = 6.957e5       # Güneş yarıçapı (km)
R_JUPITER_KM = 7.1492e4      # Jüpiter yarıçapı (km)
AU_KM        = 1.496e8       # 1 AU (km)
G            = 6.674e-11     # Gravitasyon sabiti (m³/kg/s²)
M_SUN_KG     = 1.989e30      # Güneş kütlesi (kg)


# ═════════════════════════════════════════════════════════════════
# BÖLÜM 1: MATEMATİKSEL MODELLEME — TRANSİT FİZİĞİ (0-60 dk)
# ═════════════════════════════════════════════════════════════════

def hesapla_transit_derinligi(R_p, R_s):
    """
    Transit derinliği (δ) hesabı.

    Formül:
        δ = (R_p / R_s)²

    Parametreler:
        R_p : float — Gezegen yarıçapı (herhangi bir birim)
        R_s : float — Yıldız yarıçapı (aynı birim)

    Döndürür:
        float — Transit derinliği (0 ile 1 arasında)
    """
    return (R_p / R_s) ** 2


def bolum1_transit_simulasyonu():
    """
    Bölüm 1 — Transit Geometrisi Görselleştirmesi

    Yıldızın önünden geçen bir gezegenin 5 farklı konumunu
    ve bunlara karşılık gelen ışık eğrisini yan yana gösterir.
    """
    print("\n" + "=" * 65)
    print("  BÖLÜM 1: TRANSİT FİZİĞİ VE GEOMETRİK OPTİK")
    print("=" * 65)

    # --- Parametreler ---
    R_s = 1.0           # Yıldız yarıçapı (normalize)
    R_p = 0.1           # Gezegen yarıçapı (Jüpiter ölçeğinde)

    depth = hesapla_transit_derinligi(R_p, R_s)
    print(f"\n  Yıldız Yarıçapı (R★)  : {R_s}")
    print(f"  Gezegen Yarıçapı (Rp) : {R_p}")
    print(f"  Transit Derinliği (δ) : {depth:.4f}  →  %{depth*100:.2f}")
    print(f"  Rp / R★ oranı         : {R_p/R_s:.2f}")

    # --- Gezegenin yörünge konumları ---
    pozisyonlar = np.array([-1.3, -0.8, -0.3, 0.0, 0.3, 0.8, 1.3])
    etiketler   = ['Dış (I)', 'Giriş', 'İç-I', 'Merkez', 'İç-II', 'Çıkış', 'Dış (II)']

    # --- Her konum için kapanan ışık oranını hesapla ---
    def kaplanan_oran(d, R_p, R_s):
        """
        İki dairenin kesişim alanını hesaplar.
        d : merkezler arası uzaklık
        """
        if d >= R_s + R_p:
            return 0.0
        if d + R_p <= R_s:
            return (R_p / R_s) ** 2
        if d + R_s <= R_p:
            return 1.0

        r1, r2 = R_s, R_p
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((-d+r1+r2) * (d+r1-r2) * (d-r1+r2) * (d+r1+r2))
        alan_kesisim = part1 + part2 - part3
        return alan_kesisim / (np.pi * R_s**2)

    # --- Yüksek çözünürlüklü ışık eğrisi ---
    x_yol = np.linspace(-1.5, 1.5, 1000)
    flux   = np.array([1.0 - kaplanan_oran(abs(x), R_p, R_s) for x in x_yol])

    # --- Şekil oluştur ---
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.3)

    # ÜST PANEL: Transit Geometrisi
    ax_top = fig.add_subplot(gs[0])
    theta  = np.linspace(0, 2*np.pi, 200)

    # Yıldızı çiz (gradyan efekti)
    for i in range(20, 0, -1):
        r = R_s * (i / 20)
        renk_yogunluk = 0.4 + 0.6 * (1 - (i/20)**0.5)
        daire = plt.Circle((0, 0), r, color=plt.cm.YlOrRd(renk_yogunluk),
                           alpha=0.8, zorder=1)
        ax_top.add_patch(daire)

    # Gezegeni her konumda çiz
    renkler = plt.cm.cool(np.linspace(0.2, 0.9, len(pozisyonlar)))
    for idx, (pos, etiket) in enumerate(zip(pozisyonlar, etiketler)):
        gezegen = plt.Circle((pos, 0), R_p, color=renkler[idx],
                             ec='white', lw=0.8, zorder=10, alpha=0.85)
        ax_top.add_patch(gezegen)
        ax_top.annotate(etiket, (pos, R_p + 0.08), fontsize=7,
                        ha='center', color=renkler[idx], fontweight='bold')

    ax_top.set_xlim(-1.6, 1.6)
    ax_top.set_ylim(-0.6, 0.6)
    ax_top.set_aspect('equal')
    ax_top.set_title('Transit Geometrisi — Gezegenin Yıldız Önündeki Yolculuğu',
                     fontsize=14, fontweight='bold', pad=15)
    ax_top.axhline(0, color='#30363d', ls='--', lw=0.5)
    ax_top.set_xlabel('Konum (R★ biriminde)')
    ax_top.grid(False)

    # ALT PANEL: Işık Eğrisi
    ax_bot = fig.add_subplot(gs[1])
    ax_bot.plot(x_yol, flux, color='#58a6ff', lw=2, zorder=5)
    ax_bot.fill_between(x_yol, flux, 1.0, alpha=0.15, color='#58a6ff')

    # Her konumu işaretle
    for idx, pos in enumerate(pozisyonlar):
        f_val = 1.0 - kaplanan_oran(abs(pos), R_p, R_s)
        ax_bot.plot(pos, f_val, 'o', color=renkler[idx], ms=8, zorder=10,
                    mec='white', mew=0.8)
        ax_bot.axvline(pos, color=renkler[idx], ls=':', alpha=0.3, lw=0.8)

    # Transit derinliğini göster
    ax_bot.axhline(1.0, color='#8b949e', ls='--', lw=0.8, label='Baz Çizgisi (F₀ = 1)')
    ax_bot.axhline(1.0 - depth, color='#f85149', ls='--', lw=0.8,
                   label=f'Transit Tabanı (δ = {depth:.4f})')
    ax_bot.annotate(f'δ = (Rp/R★)² = ({R_p}/{R_s})² = {depth:.4f}',
                    xy=(0, 1 - depth), xytext=(0.8, 1 - depth - 0.003),
                    fontsize=10, color='#f85149', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#f85149', lw=1.5))

    ax_bot.set_xlabel('Konum (R★ biriminde)')
    ax_bot.set_ylabel('Normalize Akı (F / F₀)')
    ax_bot.set_title('Transit Işık Eğrisi — Geometrik Model', fontsize=13, fontweight='bold')
    ax_bot.legend(loc='lower right', fontsize=9)
    ax_bot.set_ylim(1 - depth * 3, 1.001)
    ax_bot.grid(True, alpha=0.3)

    plt.savefig('bolum1_transit_geometrisi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  [✓] Grafik kaydedildi: bolum1_transit_geometrisi.png")


def bolum1_ogrenci_problemi():
    """
    Öğrenci Problemi: Farklı gezegen boyutları için transit derinliklerini karşılaştır.
    """
    print("\n  ── Öğrenci Problemi: Gezegen Boyutu vs Transit Derinliği ──")

    gezegenler = {
        'Dünya benzeri':      0.009,    # ~1 R_Earth / R_Sun
        'Süper-Dünya':        0.02,
        'Neptün benzeri':     0.035,
        'Satürn benzeri':     0.083,
        'Jüpiter benzeri':    0.10,
        'Sıcak Jüpiter':     0.15,
    }

    print(f"\n  {'Gezegen Tipi':<20} {'Rp/R★':<10} {'δ (%)':<12} {'SNR Notu':<15}")
    print("  " + "─" * 57)

    for isim, oran in gezegenler.items():
        d = hesapla_transit_derinligi(oran, 1.0)
        snr_notu = "Zor" if d < 0.001 else ("Orta" if d < 0.005 else "Kolay")
        print(f"  {isim:<20} {oran:<10.3f} {d*100:<12.4f} {snr_notu:<15}")

    # Grafik
    fig, ax = plt.subplots(figsize=(10, 6))
    oranlar = np.linspace(0.005, 0.2, 200)
    derinlikler = oranlar ** 2

    ax.plot(oranlar, derinlikler * 100, color='#58a6ff', lw=2.5)
    ax.fill_between(oranlar, derinlikler * 100, alpha=0.1, color='#58a6ff')

    renkler_g = ['#7ee787', '#3fb950', '#f0883e', '#d29922', '#f85149', '#da3633']
    for (isim, oran), renk in zip(gezegenler.items(), renkler_g):
        d = oran ** 2 * 100
        ax.plot(oran, d, 'o', color=renk, ms=10, zorder=10, mec='white', mew=1.5)
        ax.annotate(isim, (oran, d), textcoords="offset points",
                    xytext=(10, 5), fontsize=8, color=renk, fontweight='bold')

    # Tespit sınırı
    ax.axhline(0.01, color='#f85149', ls='--', lw=1, alpha=0.7,
               label='Kepler Tespit Sınırı (~100 ppm)')

    ax.set_xlabel('Rp / R★ (Yarıçap Oranı)', fontsize=12)
    ax.set_ylabel('Transit Derinliği δ (%)', fontsize=12)
    ax.set_title('Gezegen Boyutu — Transit Derinliği İlişkisi: δ = (Rp/R★)²',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.savefig('bolum1_derinlik_karsilastirma.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  [✓] Grafik kaydedildi: bolum1_derinlik_karsilastirma.png")


# ═════════════════════════════════════════════════════════════════
# BÖLÜM 2: İLERİ SİNYAL İŞLEME — LIMB DARKENING (60-120 dk)
# ═════════════════════════════════════════════════════════════════

def limb_darkening_profili(r, u1=0.4, u2=0.2):
    """
    Kuadratik Limb Darkening modeli.

    I(r) / I(0) = 1 - u1*(1-μ) - u2*(1-μ)²

    Burada μ = cos(θ) = sqrt(1 - r²), r yıldız diskindeki
    normalize yarıçap (0=merkez, 1=kenar).

    Parametreler:
        r  : float veya ndarray — Normalize yarıçap (0-1)
        u1 : float — Birinci Limb Darkening katsayısı
        u2 : float — İkinci Limb Darkening katsayısı

    Döndürür:
        float veya ndarray — Normalize yoğunluk
    """
    r = np.clip(r, 0, 0.999)
    mu = np.sqrt(1.0 - r**2)
    return 1.0 - u1 * (1 - mu) - u2 * (1 - mu)**2


def mandel_agol_transit(t, T0, P, Rp_Rs, a_Rs, inc_deg, u1=0.4, u2=0.2):
    """
    Basitleştirilmiş transit modeli (Mandel & Agol 2002 yaklaşımı).

    Kuadratik Limb Darkening ile ışık eğrisi üretir.

    Parametreler:
        t       : ndarray — Zaman dizisi (gün)
        T0      : float   — Transit merkez zamanı (gün)
        P       : float   — Yörünge periyodu (gün)
        Rp_Rs   : float   — Gezegen/Yıldız yarıçap oranı
        a_Rs    : float   — Yarı-büyük eksen / Yıldız yarıçapı
        inc_deg : float   — Yörünge eğikliği (derece)
        u1, u2  : float   — Limb darkening katsayıları

    Döndürür:
        ndarray — Normalize akı değerleri
    """
    inc = np.radians(inc_deg)
    faz = 2.0 * np.pi * (t - T0) / P
    x = a_Rs * np.sin(faz)
    y = a_Rs * np.cos(faz) * np.cos(inc)
    z = a_Rs * np.cos(faz) * np.sin(inc)

    d = np.sqrt(x**2 + y**2)   # Projeksiyon uzaklığı
    p = Rp_Rs

    flux = np.ones_like(t)

    for i in range(len(t)):
        if z[i] < 0:    # Gezegen yıldızın arkasında
            continue
        di = d[i]
        if di >= 1.0 + p:
            continue     # Transit yok

        # Sayısal integrasyon (ışık kaybı hesabı)
        n_ring = 100
        r_rings = np.linspace(0, 1.0, n_ring + 1)
        toplam_isik     = 0.0
        kaplanan_isik   = 0.0

        for j in range(n_ring):
            r_ic  = r_rings[j]
            r_dis = r_rings[j + 1]
            r_ort = (r_ic + r_dis) / 2.0
            alan  = np.pi * (r_dis**2 - r_ic**2)
            yogunluk = limb_darkening_profili(r_ort, u1, u2)
            toplam_isik += yogunluk * alan

            # Bu halkanın gezegen tarafından kaplanan kısmı
            if di + p <= r_ic or di - p >= r_dis:
                continue  # Bu halka etkilenmemiş

            # Basit yaklaşım: halkanın merkezine bakarak karar ver
            if abs(di - r_ort) < p:
                # Kaplanan alan tahmini
                theta_kap = 2.0 * np.arcsin(np.clip(p / max(r_ort, 0.01), 0, 1))
                kap_oran  = theta_kap / (2.0 * np.pi)
                kaplanan_isik += yogunluk * alan * kap_oran

        if toplam_isik > 0:
            flux[i] = 1.0 - kaplanan_isik / toplam_isik

    return flux


def bolum2_limb_darkening():
    """
    Bölüm 2 — Limb Darkening etkisi ve transit modeline katkısı.
    """
    print("\n" + "=" * 65)
    print("  BÖLÜM 2: LİMB DARKENİNG VE SİNYAL İŞLEME")
    print("=" * 65)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel 1: Limb Darkening Profilleri ──
    ax1 = fig.add_subplot(gs[0, 0])
    r = np.linspace(0, 1, 500)

    modeller = {
        'Sabit (u1=0, u2=0)':         (0.0, 0.0),
        'Lineer (u1=0.6, u2=0)':      (0.6, 0.0),
        'Kuadratik (u1=0.4, u2=0.2)': (0.4, 0.2),
        'Güçlü LD (u1=0.6, u2=0.3)':  (0.6, 0.3),
    }
    renkler_ld = ['#8b949e', '#58a6ff', '#f0883e', '#f85149']

    for (isim, (u1, u2)), renk in zip(modeller.items(), renkler_ld):
        I = limb_darkening_profili(r, u1, u2)
        ax1.plot(r, I, color=renk, lw=2, label=isim)

    ax1.set_xlabel('Normalize Yarıçap (r / R★)')
    ax1.set_ylabel('Normalize Yoğunluk I(r) / I(0)')
    ax1.set_title('Limb Darkening Profilleri', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Yıldız Diski Görünümü ──
    ax2 = fig.add_subplot(gs[0, 1])

    # 2D limb darkening görselleştirmesi
    x_grid = np.linspace(-1, 1, 500)
    X, Y = np.meshgrid(x_grid, x_grid)
    R_grid = np.sqrt(X**2 + Y**2)
    I_grid = np.where(R_grid <= 1.0, limb_darkening_profili(R_grid, 0.4, 0.2), 0)

    ax2.imshow(I_grid, extent=[-1, 1, -1, 1], cmap='YlOrRd', origin='lower',
               vmin=0, vmax=1.1)

    # Gezegen silüeti
    gezegen = plt.Circle((0.3, 0.1), 0.1, color='black', ec='white', lw=1, zorder=10)
    ax2.add_patch(gezegen)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title('Yıldız Diski + Gezegen Silüeti\n(Kuadratik LD: u₁=0.4, u₂=0.2)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('x (R★)')
    ax2.set_ylabel('y (R★)')

    # ── Panel 3: LD'li vs LD'siz Transit ──
    ax3 = fig.add_subplot(gs[1, 0])

    # WASP-126 b benzeri parametreler
    P     = 3.2888        # Periyot (gün)
    T0    = 0.0           # Transit merkezi
    Rp_Rs = 0.078         # Yarıçap oranı (NASA Exoplanet Archive)
    a_Rs  = 7.8           # Yarı-büyük eksen / R★
    inc   = 87.8          # Eğiklik (derece)

    t_transit = np.linspace(-0.15, 0.15, 800)

    # LD'siz model (düz taban)
    flux_flat = mandel_agol_transit(t_transit, T0, P, Rp_Rs, a_Rs, inc, u1=0, u2=0)
    # LD'li model (kavisli taban)
    flux_ld   = mandel_agol_transit(t_transit, T0, P, Rp_Rs, a_Rs, inc, u1=0.4, u2=0.2)

    ax3.plot(t_transit * 24, flux_flat, color='#8b949e', lw=2,
             ls='--', label='LD Yok (düz taban)', alpha=0.8)
    ax3.plot(t_transit * 24, flux_ld, color='#f0883e', lw=2.5,
             label='Kuadratik LD (u₁=0.4, u₂=0.2)')

    ax3.set_xlabel('Transit Merkezinden Uzaklık (saat)')
    ax3.set_ylabel('Normalize Akı')
    ax3.set_title('Limb Darkening Etkisi: Düz vs Kavisli Taban',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Fark bölgesini vurgula
    ax3.fill_between(t_transit * 24, flux_flat, flux_ld,
                     alpha=0.2, color='#f0883e', label='LD Farkı')

    # ── Panel 4: Savitzky-Golay Filtresi ile Gürültü Temizleme ──
    ax4 = fig.add_subplot(gs[1, 1])

    # Sentetik gürültülü veri üret (öğretim amaçlı simülasyon)
    np.random.seed(42)
    t_data = np.linspace(-0.2, 0.2, 600)
    flux_temiz = mandel_agol_transit(t_data, T0, P, Rp_Rs, a_Rs, inc, 0.4, 0.2)
    gurultu = np.random.normal(0, 0.0015, len(t_data))
    flux_ham = flux_temiz + gurultu

    # Savitzky-Golay filtresi
    flux_sg = savgol_filter(flux_ham, window_length=31, polyorder=3)

    ax4.scatter(t_data * 24, flux_ham, s=2, alpha=0.4, color='#8b949e',
                label='Ham Veri (gürültülü)')
    ax4.plot(t_data * 24, flux_sg, color='#f85149', lw=2,
             label='Savitzky-Golay Filtresi')
    ax4.plot(t_data * 24, flux_temiz, color='#7ee787', lw=1.5, ls='--',
             alpha=0.7, label='Gerçek Sinyal')

    ax4.set_xlabel('Transit Merkezinden Uzaklık (saat)')
    ax4.set_ylabel('Normalize Akı')
    ax4.set_title('Sinyal İşleme: Savitzky-Golay Filtresi',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)

    plt.savefig('bolum2_limb_darkening.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  [✓] Grafik kaydedildi: bolum2_limb_darkening.png")

    # İstatistiksel karşılaştırma
    residual_ham = flux_ham - flux_temiz
    residual_sg  = flux_sg  - flux_temiz
    print(f"\n  Sinyal İşleme Sonuçları:")
    print(f"  ├─ Ham veri RMS hatası       : {np.std(residual_ham)*1e6:.0f} ppm")
    print(f"  ├─ SG filtreli RMS hatası    : {np.std(residual_sg)*1e6:.0f} ppm")
    print(f"  └─ İyileşme faktörü          : {np.std(residual_ham)/np.std(residual_sg):.1f}x")


# ═════════════════════════════════════════════════════════════════
# BÖLÜM 3: FİZİKSEL KARAKTERİZASYON — KEPLER YASALARI (120-180 dk)
# ═════════════════════════════════════════════════════════════════

def bolum3_kepler_ve_karakterizasyon():
    """
    Bölüm 3 — Kepler Yasaları ile fiziksel parametrelerin hesaplanması.
    WASP-126 b örneği üzerinden temsili karakterizasyon (öğretim amaçlı simülasyon).
    """
    print("\n" + "=" * 65)
    print("  BÖLÜM 3: KEPLER YASALARI VE FİZİKSEL KARAKTERİZASYON")
    print("=" * 65)

    # ── WASP-126 b Parametreleri ──
    yildiz_kutle  = 1.12          # M★ (Güneş kütlesi cinsinden)
    yildiz_yaricap = 1.27         # R★ (Güneş yarıçapı cinsinden)
    yildiz_sicaklik = 5800        # T_eff (K)
    periyot       = 3.2888        # P (gün)
    Rp_Rs         = 0.078         # Rp/R★ (NASA Exoplanet Archive)
    transit_suresi = 3.42          # Transit süresi (saat)

    print(f"\n  ═══ WASP-126 Sistemi Verileri ═══")
    print(f"  Yıldız Kütlesi    : {yildiz_kutle} M☉")
    print(f"  Yıldız Yarıçapı   : {yildiz_yaricap} R☉")
    print(f"  Yıldız Sıcaklığı  : {yildiz_sicaklik} K")
    print(f"  Yörünge Periyodu  : {periyot} gün")

    # ── Hesap 1: Gezegen Yarıçapı ──
    R_p_Rsun = Rp_Rs * yildiz_yaricap
    R_p_Rjup = R_p_Rsun * R_SUN_KM / R_JUPITER_KM
    R_p_km   = R_p_Rsun * R_SUN_KM

    print(f"\n  ── Hesap 1: Gezegen Yarıçapı ──")
    print(f"  Rp = (Rp/R★) × R★ = {Rp_Rs} × {yildiz_yaricap} R☉")
    print(f"  Rp = {R_p_Rsun:.3f} R☉ = {R_p_Rjup:.2f} R_J = {R_p_km:.0f} km")

    # ── Hesap 2: Yörünge Yarıçapı (Kepler 3. Yasa) ──
    # a³ = (G × M★ × P²) / (4π²)
    P_saniye = periyot * 86400.0
    M_kg     = yildiz_kutle * M_SUN_KG
    a_metre  = (G * M_kg * P_saniye**2 / (4 * np.pi**2)) ** (1/3)
    a_au     = a_metre / (AU_KM * 1e3)
    a_km     = a_metre / 1e3

    # Basitleştirilmiş formül ile karşılaştırma
    a_au_basit = (yildiz_kutle * (periyot / 365.25)**2) ** (1/3)

    print(f"\n  ── Hesap 2: Yörünge Yarıçapı (Kepler 3. Yasa) ──")
    print(f"  a³ = G × M★ × P² / (4π²)")
    print(f"  a  = {a_au:.4f} AU = {a_km:.0f} km")
    print(f"  a  ≈ {a_au_basit:.4f} AU (basit formül)")

    # ── Hesap 3: Denge Sıcaklığı ──
    albedo = 0.3   # Varsayılan albedo (Bond)
    T_eq   = yildiz_sicaklik * np.sqrt(yildiz_yaricap * R_SUN_KM / (2 * a_km)) \
             * (1 - albedo)**0.25

    print(f"\n  ── Hesap 3: Denge Sıcaklığı ──")
    print(f"  T_eq = T★ × √(R★ / 2a) × (1-A)^(1/4)")
    print(f"  T_eq = {T_eq:.0f} K  (Albedo = {albedo})")
    print(f"  Karşılaştırma: Dünya ≈ 255 K, Venüs ≈ 230 K, Merkür ≈ 440 K")

    # ── Hesap 4: Yaşanabilirlik Bölgesi Kontrolü ──
    # Lüminozite temelli HZ hesabı (Kopparapu et al. 2013)
    # WASP-126: log10(L★/L☉) ≈ 0.145  →  L★ ≈ 1.40 L☉
    L_star = 10**0.145  # L☉ cinsinden (NASA Exoplanet Archive)
    hz_ic  = 0.95 * np.sqrt(L_star)    # AU (iç sınır, kaçak sera limiti)
    hz_dis = 1.67 * np.sqrt(L_star)    # AU (dış sınır, maksimum sera limiti)

    print(f"\n  ── Hesap 4: Yaşanabilirlik Bölgesi ──")
    print(f"  HZ İç Sınır  : {hz_ic:.3f} AU")
    print(f"  HZ Dış Sınır : {hz_dis:.3f} AU")
    print(f"  Gezegen       : {a_au:.3f} AU  →  {'HZ İÇİNDE ✓' if hz_ic < a_au < hz_dis else 'HZ DIŞINDA ✗'}")

    # ── Büyük Görselleştirme ──
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Kepler 3. Yasa Grafiği
    ax1 = fig.add_subplot(gs[0, 0])
    P_range = np.linspace(0.5, 400, 500)  # gün
    a_range = (yildiz_kutle * (P_range / 365.25)**2) ** (1/3)

    ax1.plot(P_range, a_range, color='#58a6ff', lw=2)
    ax1.plot(periyot, a_au, '*', color='#f0883e', ms=20, zorder=10,
             mec='white', mew=1.5, label=f'WASP-126 b\n(P={periyot}d, a={a_au:.3f}AU)')

    # Güneş sistemi gezegenlerini ekle
    ss_gezegenler = {
        'Merkür':   (87.97,  0.387),
        'Venüs':    (224.7,  0.723),
        'Dünya':    (365.25, 1.000),
        'Mars':     (687.0,  1.524),
    }
    for isim, (p_g, a_g) in ss_gezegenler.items():
        if p_g <= 400:
            ax1.plot(p_g, a_g, 'o', color='#7ee787', ms=8, mec='white', mew=1)
            ax1.annotate(isim, (p_g, a_g), textcoords="offset points",
                         xytext=(8, 5), fontsize=8, color='#7ee787')

    ax1.axhspan(hz_ic, hz_dis, alpha=0.15, color='#3fb950', label='Yaşanabilirlik Bölgesi')
    ax1.set_xlabel('Yörünge Periyodu (gün)')
    ax1.set_ylabel('Yarı-büyük Eksen (AU)')
    ax1.set_title('Kepler 3. Yasası: P² ∝ a³', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Panel 2: BLS Periodogram Simülasyonu
    ax2 = fig.add_subplot(gs[0, 1])

    test_periyotlar = np.linspace(1.0, 10.0, 5000)
    np.random.seed(123)
    bls_gucu = np.random.exponential(0.5, len(test_periyotlar))

    # Gerçek periyotta tepe ekle
    idx_gercek = np.argmin(np.abs(test_periyotlar - periyot))
    bls_gucu[idx_gercek-5:idx_gercek+5] += np.array([2, 5, 10, 18, 25, 25, 18, 10, 5, 2])

    # Harmoniklerde küçük tepeler
    for harmonik in [periyot/2, periyot*2]:
        idx_h = np.argmin(np.abs(test_periyotlar - harmonik))
        bls_gucu[max(0,idx_h-3):idx_h+3] += np.array([1, 3, 6, 6, 3, 1])

    ax2.plot(test_periyotlar, bls_gucu, color='#58a6ff', lw=0.8, alpha=0.8)
    ax2.axvline(periyot, color='#f85149', ls='--', lw=1.5,
                label=f'En güçlü periyot: {periyot:.4f} gün')
    ax2.axhline(7.0, color='#f0883e', ls=':', lw=1,
                label='SNR Eşiği (7σ)', alpha=0.7)

    ax2.set_xlabel('Test Periyodu (gün)')
    ax2.set_ylabel('BLS Gücü (σ)')
    ax2.set_title('BLS Periodogram — Temsili Simülasyon', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Yörünge Diyagramı
    ax3 = fig.add_subplot(gs[1, 0])

    theta_orbit = np.linspace(0, 2*np.pi, 500)
    x_orbit = a_au * np.cos(theta_orbit)
    y_orbit = a_au * np.sin(theta_orbit)

    # Yıldız
    yildiz_patch = plt.Circle((0, 0), yildiz_yaricap * R_SUN_KM / AU_KM,
                               color='#f0883e', alpha=0.9, zorder=5)
    ax3.add_patch(yildiz_patch)

    # Yörünge
    ax3.plot(x_orbit, y_orbit, color='#58a6ff', lw=1.5, ls='--', alpha=0.6)

    # Gezegen
    gez_x = a_au * np.cos(np.pi/4)
    gez_y = a_au * np.sin(np.pi/4)
    gez_patch = plt.Circle((gez_x, gez_y), 0.003, color='#7ee787', zorder=10)
    ax3.add_patch(gez_patch)
    ax3.annotate('WASP-126 b', (gez_x, gez_y), textcoords="offset points",
                 xytext=(10, 10), fontsize=10, color='#7ee787', fontweight='bold')

    # Yaşanabilirlik Bölgesi
    hz_ic_circle  = plt.Circle((0, 0), hz_ic, fill=False, color='#3fb950',
                                ls='--', lw=1, alpha=0.5)
    hz_dis_circle = plt.Circle((0, 0), hz_dis, fill=False, color='#3fb950',
                                ls='--', lw=1, alpha=0.5)
    ax3.add_patch(hz_ic_circle)
    ax3.add_patch(hz_dis_circle)
    ax3.annotate('HZ', (hz_dis * 0.7, hz_dis * 0.7), fontsize=9,
                 color='#3fb950', alpha=0.7)

    ax3.set_xlim(-0.15, 0.15)
    ax3.set_ylim(-0.15, 0.15)
    ax3.set_aspect('equal')
    ax3.set_xlabel('x (AU)')
    ax3.set_ylabel('y (AU)')
    ax3.set_title('Yörünge Diyagramı', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.2)

    # Panel 4: Gezegen Kimlik Kartı
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Sınıflandırma
    if R_p_Rjup > 0.8:
        gezegen_sinif = "Sıcak Gaz Devi (düşük yoğunluklu)"
    elif R_p_Rjup > 0.3:
        gezegen_sinif = "Sıcak Satürn"
    elif R_p_Rjup > 0.15:
        gezegen_sinif = "Sıcak Neptün"
    else:
        gezegen_sinif = "Süper-Dünya"

    kimlik_karti = f"""
    ╔═══════════════════════════════════════╗
    ║     GEZEGEN KİMLİK KARTI             ║
    ║     WASP-126 b                        ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║  Yarıçap    : {R_p_Rjup:.2f} R_J ({R_p_km:.0f} km)  ║
    ║  Periyot    : {periyot:.4f} gün             ║
    ║  Uzaklık    : {a_au:.4f} AU               ║
    ║  Sıcaklık   : {T_eq:.0f} K                  ║
    ║  Sınıf      : {gezegen_sinif:<22}║
    ║  HZ Durumu  : {'İÇİNDE' if hz_ic < a_au < hz_dis else 'DIŞINDA':<22}║
    ║  SNR        : > 7σ (öğretim eşiği) ✓  ║
    ║  V-Shape    : Taban düz ✓             ║
    ║                                       ║
    ╠═══════════════════════════════════════╣
    ║  DURUM: TEMSİLİ DOĞRULAMA ✓          ║
    ╚═══════════════════════════════════════╝
    """

    ax4.text(0.05, 0.95, kimlik_karti, fontsize=10, fontfamily='monospace',
             verticalalignment='top', color='#7ee787',
             bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d'))
    ax4.set_title('Teknik Karakterizasyon Raporu (Öğretim Amaçlı)', fontsize=12, fontweight='bold')

    plt.savefig('bolum3_kepler_karakterizasyon.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  [✓] Grafik kaydedildi: bolum3_kepler_karakterizasyon.png")

    return {
        'R_p_Rjup': R_p_Rjup,
        'periyot': periyot,
        'a_au': a_au,
        'T_eq': T_eq,
        'sinif': gezegen_sinif,
        'hz_icinde': hz_ic < a_au < hz_dis,
    }


def bolum3_karar_matrisi(sonuclar):
    """
    Bölüm 3 — Teknik Kontrol Listesi (Karar Matrisi).
    """
    print("\n  ── TEKNİK KONTROL LİSTESİ (Öğretim Amaçlı Doğrulama) ──\n")

    kontroller = [
        ('SNR',              '> 7.0 σ',           True,
         'Sinyal gürültüden yeterince güçlü. (Öğretim eşiği; Kepler hattında 7.1 MES kullanılırdı.)'),
        ('V-Shape Testi',    'Taban düz (U)',      True,
         'U şekli → Gezegen. V olsaydı → Çift yıldız.'),
        ('Secondary Eclipse', 'İkinci çukur yok',  True,
         'İkinci çukur yok → Çift yıldız değil.'),
        ('Duration Check',   'Süre uyumlu',        True,
         f'P={sonuclar["periyot"]:.2f}d, a={sonuclar["a_au"]:.3f}AU ile tutarlı.'),
        ('Periyot Stabilitesi', '±0.001 gün içi',  True,
         'Ardışık transitler arası periyot sabit.'),
    ]

    print(f"  {'Kontrol':<22} {'Kriter':<20} {'Sonuç':<12} {'Fiziksel Yorum'}")
    print("  " + "─" * 85)
    for isim, kriter, basarili, yorum in kontroller:
        durum = '✓ Başarılı' if basarili else '✗ Başarısız'
        renk_kod = '' if basarili else '  ⚠️'
        print(f"  {isim:<22} {kriter:<20} {durum:<12} {yorum}{renk_kod}")

    hepsi_basarili = all(k[2] for k in kontroller)
    print("\n  " + "═" * 85)
    if hepsi_basarili:
        print("  ✅  TÜM KONTROLLER BAŞARILI → TEMSİLİ DOĞRULAMA TAMAMLANDI (Öğretim amaçlı)")
    else:
        print("  ❌  BAZI KONTROLLER BAŞARISIZ → TEMSİLİ DOĞRULAMA BAŞARISIZ")
    print("  " + "═" * 85)


# ═════════════════════════════════════════════════════════════════
# ANA PROGRAM
# ═════════════════════════════════════════════════════════════════

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🔭  ASTRO-MÜHENDİSLİK VE FİZİKSEL MODELLEME AKADEMİSİ   ║
    ║       Ötegezegen Transit Analizi Atölye Programı             ║
    ║                                                              ║
    ║   Hedef Sistem : WASP-126 b                                  ║
    ║   Süre         : 3 × 60 dakika                               ║
    ║   Seviye       : Lise (İleri Düzey)                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── BÖLÜM 1 ──
    bolum1_transit_simulasyonu()
    bolum1_ogrenci_problemi()

    # ── BÖLÜM 2 ──
    bolum2_limb_darkening()

    # ── BÖLÜM 3 ──
    sonuclar = bolum3_kepler_ve_karakterizasyon()
    bolum3_karar_matrisi(sonuclar)

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ATÖLYE TAMAMLANDI                                          ║
    ║                                                              ║
    ║   Üretilen Grafikler:                                        ║
    ║   ├─ bolum1_transit_geometrisi.png                           ║
    ║   ├─ bolum1_derinlik_karsilastirma.png                       ║
    ║   ├─ bolum2_limb_darkening.png                               ║
    ║   └─ bolum3_kepler_karakterizasyon.png                       ║
    ║                                                              ║
    ║   Her öğrenciden beklenen çıktı:                             ║
    ║   → Gezegen Teknik Karakterizasyon Raporu                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
