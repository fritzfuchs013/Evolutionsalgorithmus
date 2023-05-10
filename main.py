import random
import copy as cp
import numpy as np                          # Paket für numerische Operationen
import matplotlib.pyplot as plt             # Paket fürs grafische Darstellen
from matplotlib import rc
import math

"""
Optimierung Übung 3

• 1-gliedrig
• mu,lambda-Strategie
• 2d Zielfunktion

Beispiel booth Funktion https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

# Anpassung Schriftart & -größe für Plots
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.rcParams['figure.figsize'] = [5, 3]     # Anpassung der Plot-Größe


# Definition der Optimierungsaufgabe
def zf(xVec):
    """
    Definiton Zielfunktion

    :param xVec: Punkt im Eingangsraum
    :return: zugehörigen Zielfunktionswert
    """

    # z = (xVec[0]+2*xVec[1]-7)**2 + (2*xVec[0]+xVec[1]-5)**2
    z = (math.sin(xVec[0]) + 7 * (math.sin(xVec[1]))**2 + 0.1 * (xVec[2])**4 * math.sin(xVec[0]))
    return z


# Methoden
def suchBereichBestimmen(x, maxD, c1, c2):
    """
    Bestimmung des Suchbereiches (Bereich in dem Nachkommen generiert werden)

    :param x:
    :param maxD:
    :param c1: Größe der Maximalbox
    :param c2: Größe der Minimalbox
    :return:
    """

    maxSB = np.zeros(shape=(3, 2))
    minSB = np.zeros(shape=(3, 2))
    maxSB[:, 0] = x - (0.5 * maxD * c1)
    maxSB[:, 1] = x + (0.5 * maxD * c1)
    minSB[:, 0] = x - (0.5 * maxD * c2)
    minSB[:, 1] = x + (0.5 * maxD * c2)

    return maxSB, minSB


def getNachkomme(minSB, maxSB):
    """
    Generiere Nachkomme innerhalb vordefiniertem Suchbereich

    :param minSB:
    :param maxSB:
    :return: Koordinaten Nachkomme (im Eingangsraum)
    """

    # Vektoren (für 3 dimensionalen Eingangsraum!)
    xSearch = np.zeros(shape=(2, 3))
    xNachkomme = np.zeros(shape=(3,))

    for idx in range(3):
        xSearch[idx, 0] = random.uniform(maxSB[idx, 0], minSB[idx, 0])
        xSearch[idx, 1] = random.uniform(minSB[idx, 1], maxSB[idx, 1])
        xSearch[idx, 2] = random.uniform(minSB[idx, 2], maxSB[idx, 2])

    # Zufällige Wahl eines Nachkommens
    for idx in range(2):
        xNachkomme[idx] = xSearch[idx, random.randint(0, 1)]

    return xNachkomme


def checkNB(xVec, nb):
    """
    Prüfung der Nebenbedingungen, bei Verletzung der Nebenbedingung wird der Punkt auf NB zurückgeschoben

    :param xVec: Vektor im Eingangsraum
    :param nb: Nebenbedingung
    :return: (korrigierter) Vektor im Eingangsraum
    """

    for idx in range(2):
        if xVec[idx] < nb[idx, 0]:
            xVec[idx] = cp.deepcopy(nb[idx, 0])
        if xVec[idx] > nb[idx, 1]:
            xVec[idx] = cp.deepcopy(nb[idx, 1])

    return xVec


# Ergebnisvisualisierung
def plotResults(zfHistory, xHistory):
    """

    :param zfHistory:
    :param xHistory:
    :return: None
    """

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(zfHistory, 'rx-', markersize=5, linewidth=1)
    ax[0].set_yscale('log')
    ax[1].plot(xHistory[:, 0], xHistory[:, 1], 'bx-', markersize=5, linewidth=1)
    ax[1].plot(1, 3, 'r.', markersize=3)
    ax[0].grid()
    ax[1].grid()
    ax[0].set_title('Optimierungsfortschritt')
    ax[0].set_ylabel('Zielfunktionswert')
    ax[0].set_xlabel('Optimierungsiteration')
    ax[1].set_title('Eingangsraumabsuche')
    ax[1].set_ylabel(r'$x_2$')
    ax[1].set_xlabel(r'$x_1$')
    fig.tight_layout()

    plt.show()


# Hauptprogramm

# Definition der Nebenbedingungen (zulässiger Bereich)
nb = np.ndarray(shape=(3, 2))
nb[0, :] = [-math.pi, math.pi]        # Nebenbedingung in x1 Richtung
nb[1, :] = [-math.pi, math.pi]        # Nebenbedingung in x2 Richtung
nb[2, :] = [-math.pi, math.pi]        # Nebenbedingung in x3 Richtung

# Definition des maximalen Suchbereiches
maxD = np.ndarray(shape=(3,))
maxD[0] = math.pi    # Suchbereich in x1 Richtung
maxD[1] = math.pi     # Suchbereich in x2 Richtung
maxD[1] = math.pi     # Suchbereich in x3 Richtung

# Definition der Optimierungsparameter
c1 = 0.8
c2 = 0.1
c5 = 0.3
npAnz = 5
npCt = 0
runAnz = 45     # Anzahl der Iterationen

# Initialer Startpunkt
x = np.ndarray(shape=(3,))
x[0] = random.uniform(nb[0, 0], nb[0, 1])
x[1] = random.uniform(nb[1, 0], nb[1, 1])
x[2] = random.uniform(nb[2, 0], nb[2, 1])

# Optimierungsschleife
zfHistory = []
xHistory = np.zeros(shape=(runAnz, 2))

for idx in range(runAnz):
    print('Iteration: %i' %idx)
    # Schrittweiten/Suchbereichsdefinition
    maxSB, minSB = suchBereichBestimmen(x, maxD, c1, c2)
    # Zielfunktionsauswertung
    zfAlt = zf(x)
    # Erzeugen eines Nachkommens
    xNeu = getNachkomme(minSB,maxSB)
    # Berechnung des Suchvektors
    suchVec = cp.deepcopy(xNeu-x)

    # Vergleich Zielfunktion
    if zf(xNeu) < zfAlt:
        print('Fall 1: neuer Zielfunktionswert ist besser als alter, Kind wird neuer Elter')
        x = cp.deepcopy(checkNB(xNeu, nb))
        npCt = 0

    elif zf(x - suchVec) < zfAlt:
        print('Fall 2: Suche in entgegengesetzer Suchrichtung, "Spiegelkind" wird neuer Elter')
        x = cp.deepcopy(checkNB(x - suchVec, nb))
        npCt = 0

    elif npCt >= npAnz:
        print('Fall 4: falls npAnz negative Versuche erfolgt sind, wird Suchbereich verringert')
        maxD = cp.deepcopy(c5 * maxD)
        npCt = 0
    else:
        print('Fall 3: keine Verbesserung, nächster Versuch')
        npCt += 1
        pass

    print(''.join(['ZF-Wert %e, Suchpunkt:' % (zf(x)), str(x)]))
    zfHistory.append(zf(x))
    xHistory[idx, :] = cp.deepcopy(x)

plotResults(zfHistory, xHistory)
