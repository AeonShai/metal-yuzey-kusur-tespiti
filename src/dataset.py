"""MVTec AD veri setini okumak için yardımcı fonksiyonlar.

Klasör yapısı (her kategori için):
    <kategori>/train/good/*.png              -> yalnızca sağlam görüntüler
    <kategori>/test/<tür>/*.png              -> sağlam (good) ve kusurlu görüntüler
    <kategori>/ground_truth/<tür>/*_mask.png -> kusurlu görüntülerin piksel maskeleri
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJE_KOKU = Path(__file__).resolve().parent.parent
VERI_KOKU = PROJE_KOKU / "data" / "mvtec_ad"


@dataclass
class Ornek:
    """Veri setindeki tek bir görüntü."""

    goruntu_yolu: Path
    bolum: str            # "train" veya "test"
    kusur_turu: str       # "good", "scratch", "bent", "color", "flip"
    maske_yolu: Path | None

    @property
    def kusurlu_mu(self) -> bool:
        return self.kusur_turu != "good"

    @property
    def ad(self) -> str:
        return f"{self.bolum}/{self.kusur_turu}/{self.goruntu_yolu.stem}"


def ornekleri_listele(kategori: str = "metal_nut", bolum: str = "test",
                      veri_koku: Path = VERI_KOKU) -> list[Ornek]:
    """Bir kategori ve bölümdeki tüm görüntüleri sıralı biçimde döndürür."""
    bolum_klasoru = veri_koku / kategori / bolum
    if not bolum_klasoru.is_dir():
        raise FileNotFoundError(
            f"{bolum_klasoru} bulunamadı. Veri setinin data/mvtec_ad altına "
            "açıldığından emin olun (README'deki adımlara bakın)."
        )

    ornekler = []
    for tur_klasoru in sorted(p for p in bolum_klasoru.iterdir() if p.is_dir()):
        tur = tur_klasoru.name
        for goruntu_yolu in sorted(tur_klasoru.glob("*.png")):
            maske_yolu = None
            if tur != "good":
                maske_yolu = (veri_koku / kategori / "ground_truth" / tur
                              / f"{goruntu_yolu.stem}_mask.png")
            ornekler.append(Ornek(goruntu_yolu, bolum, tur, maske_yolu))
    return ornekler


def goruntu_oku(ornek: Ornek, gri: bool = False) -> np.ndarray:
    """Görüntüyü okur. Renkli okunduğunda kanal sırası RGB'dir (matplotlib ile uyumlu)."""
    bayrak = cv2.IMREAD_GRAYSCALE if gri else cv2.IMREAD_COLOR
    # Türkçe karakterli yollarda cv2.imread çalışmayabildiği için baytlardan okunur.
    veri = np.fromfile(str(ornek.goruntu_yolu), dtype=np.uint8)
    goruntu = cv2.imdecode(veri, bayrak)
    if goruntu is None:
        raise IOError(f"Görüntü okunamadı: {ornek.goruntu_yolu}")
    if not gri:
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
    return goruntu


def maske_oku(ornek: Ornek) -> np.ndarray:
    """Referans maskeyi 0/1 değerli bool dizi olarak döndürür.

    Sağlam görüntülerin maskesi yoktur; bunlar için tamamen boş bir maske döner.
    """
    if ornek.maske_yolu is None:
        boyut = goruntu_oku(ornek, gri=True).shape
        return np.zeros(boyut, dtype=bool)
    veri = np.fromfile(str(ornek.maske_yolu), dtype=np.uint8)
    maske = cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE)
    if maske is None:
        raise IOError(f"Maske okunamadı: {ornek.maske_yolu}")
    return maske > 127
