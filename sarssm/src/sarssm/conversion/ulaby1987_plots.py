import numpy as np
import matplotlib.pyplot as plt
import fsarcamp as fc
import ulaby1987


def plot_corn_dielectrics_ulaby():
    m_g = np.linspace(0.05, 0.7, 100)
    for band in ["X", "C", "L"]:
        frequency_hz = fc.get_fsar_center_frequency(band)
        eps = ulaby1987.corn_moisture_to_eps_ulaby(m_g, frequency_hz)
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.set_title(f"Corn leaf dielectrics, Ulaby model\n{band}-band, {frequency_hz * 1e-9:.3f} GHz")
        ax.plot(m_g, eps.real, label="real")
        ax.plot(m_g, np.abs(eps.imag), label="|imag|")
        ax.set_xlabel("Gravimetric moisture")
        ax.set_ylabel("Dielectric constant")
        ax.set_xlim(0.05, 0.7)
        ax.set_ylim(0, 35)
        ax.legend()
        ax.grid()
        fig.savefig(f"visualization/ulaby_corn_dielectrics_{band}_band.png", dpi=400)
        plt.close("all")


if __name__ == "__main__":
    plot_corn_dielectrics_ulaby()
