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
def zf(xVec,optimierungsziel,ketten,nachkommen,start):
    """
    Definiton Zielfunktion

    :param xVec: Punkt im Eingangsraum
    :return: zugehörigen Zielfunktionswert
    """
    zSpeicher = np.zeros(shape=(nachkommen, 1, ketten))

    if start:
        for i in range(ketten):
            z = (math.sin(xVec[0, 0, i]) + 7 * (math.sin(xVec[0, 1, i])) ** 2 + 0.1 * (xVec[0,2,i]) ** 4 * math.sin(xVec[0, 0, i]))
            if optimierungsziel:
                zSpeicher[0, 0, i] = z
            else:
                zSpeicher[0, 0, i] = z * (-1)

    else:
        for i in range(ketten):
            for j in range(nachkommen):
                z = (math.sin(xVec[j,0,i]) + 7 * (math.sin(xVec[j,1,i]))**2 + 0.1 * (xVec[j,2,i])**4 * math.sin(xVec[j,0,i]))
                if optimierungsziel:
                    zSpeicher[j,0,i] = z
                else:
                    zSpeicher[j,0,i] = z*(-1)

    return zSpeicher

# Methoden
def suchBereichBestimmen(x, maxD, c1, c2, ketten):
    """
    Bestimmung des Suchbereiches (Bereich in dem Nachkommen generiert werden)

    :param x:
    :param maxD:
    :param c1: Größe der Maximalbox
    :param c2: Größe der Minimalbox
    :return:
    """

    maxSB = np.zeros(shape=(2,3,ketten))
    minSB = np.zeros(shape=(2,3,ketten))
    for i in range(ketten):
        print(str(x))
        maxSB[0, :, i] = x[0,:,i] - (0.5 * maxD * c1)
        maxSB[1, :, i] = x[0,:,i] + (0.5 * maxD * c1)
        minSB[0, :, i] = x[0,:,i] - (0.5 * maxD * c2)
        minSB[1, :, i] = x[0,:,i] + (0.5 * maxD * c2)

    return maxSB, minSB


def getNachkomme(minSB, maxSB,ketten,nachkommen):
    """
    Generiere Nachkomme innerhalb vordefiniertem Suchbereich

    :param minSB:
    :param maxSB:
    :return: Koordinaten Nachkomme (im Eingangsraum)
    """

    # Vektoren (für 3 dimensionalen Eingangsraum!)
    xNachkomme = np.zeros(shape=(nachkommen, 3, ketten))

    for i in range(ketten):
        for j in range(nachkommen):
            xNachkomme[j, 0, i] = random.uniform(maxSB[0, 0, i], minSB[1, 0, i])
            xNachkomme[j, 1, i] = random.uniform(minSB[0, 1, i], maxSB[1, 1, i])
            xNachkomme[j, 2, i] = random.uniform(minSB[0, 2, i], maxSB[1, 2, i])


    return xNachkomme


def checkNB(xVec, nb, ketten, nachkommen):
    """
    Prüfung der Nebenbedingungen, bei Verletzung der Nebenbedingung wird der Punkt auf NB zurückgeschoben

    :param xVec: Vektor im Eingangsraum
    :param nb: Nebenbedingung
    :return: (korrigierter) Vektor im Eingangsraum
    """

    for i in range(ketten):
        for j in range(nachkommen):
            for k in range(3):
                if xVec[j, k, i] < nb[k, 0]:
                    xVec[j, k, i] = cp.deepcopy(nb[k, 0])
                if xVec[j, k, i] > nb[k, 1]:
                    xVec[j, k, i] = cp.deepcopy(nb[k, 1])

    return xVec


# Ergebnisvisualisierung
# def plotResults(zfHistory, xHistory):
#     """
#
#     :param zfHistory:
#     :param xHistory:
#     :return: None
#     """
#
#     fig, ax = plt.subplots(1, 2)
#     ax[0].plot(zfHistory, 'rx-', markersize=5, linewidth=1)
#     ax[0].set_yscale('log')
#     ax[1].plot(xHistory[:, 0], xHistory[:, 1], 'bx-', markersize=5, linewidth=1)
#     ax[1].plot(1, 3, 'r.', markersize=3)
#     ax[0].grid()
#     ax[1].grid()
#     ax[0].set_title('Optimierungsfortschritt')
#     ax[0].set_ylabel('Zielfunktionswert')
#     ax[0].set_xlabel('Optimierungsiteration')
#     ax[1].set_title('Eingangsraumabsuche')
#     ax[1].set_ylabel(r'$x_2$')
#     ax[1].set_xlabel(r'$x_1$')
#     fig.tight_layout()
#
#     plt.show()


# Hauptprogramm
# optimierungsziel ist true(min) oder false(max -> zf * (-1))
# Anzahl an Iterationen
# Ketten Anzahl (wieviele Startpunkte)
# Nachkommen = Wieviel Kinder je Kette (vielfaches von 5), nur mit dem besten Weiter
def run_optimierung(optimierungsziel,iterationen,ketten,nachkommen):

    if nachkommen % 5 > 0:
        print('Die Anzahl der Nachkommen muss ein Vielfaches von 5 sein!')
        exit(1)

    zfAlt = np.ndarray(shape=(1,2,ketten))

    # Definition der Nebenbedingungen (zulässiger Bereich)
    nb = np.ndarray(shape=(3, 2))
    nb[0, :] = [-math.pi, math.pi]        # Nebenbedingung in x1 Richtung
    nb[1, :] = [-math.pi, math.pi]        # Nebenbedingung in x2 Richtung
    nb[2, :] = [-math.pi, math.pi]        # Nebenbedingung in x3 Richtung

    # Definition des maximalen Suchbereiches
    # maxD = np.ndarray(shape=(2,3,ketten))
    # maxD[:,0,:] = math.pi    # Suchbereich in x1 Richtung
    # maxD[:,1,:] = math.pi     # Suchbereich in x2 Richtung
    # maxD[:,2,:] = math.pi     # Suchbereich in x3 Richtung
    maxD = math.pi

    # Definition der Optimierungsparameter
    c1 = 0.8
    c2 = 0.1
    c5 = 0.85
    c6 = 1.15
    sbKleiner = 10*ketten
    sbKleinerCounter = 0
    zfverbesserungen = 0


    # Initiale Startpunkte
    start = True
    x = np.ndarray(shape=(1, 3, ketten))
    for i in range(ketten):
        x[0,0,i] = random.uniform(nb[0, 0], nb[0, 1])
        x[0,1,i] = random.uniform(nb[1, 0], nb[1, 1])
        x[0,2,i] = random.uniform(nb[2, 0], nb[2, 1])

    # Optimierungsschleife
    zfHistory = []
    xHistory = np.zeros(shape=(iterationen, 3, ketten))

    for it in range(iterationen):
        print('Iteration: %i' %it)
        # Schrittweiten/Suchbereichsdefinition
        maxSB, minSB = suchBereichBestimmen(x, maxD, c1, c2, ketten)
        # Zielfunktionsauswertung
        if start:
            for i in range(ketten):
                zfAlt[0,0,i] = zf(x,optimierungsziel,ketten,nachkommen,start)[0,0,i]
                zfAlt[0,1,i] = 1

        start = False
        # Erzeugen eines Nachkommens
        xNeu = getNachkomme(minSB,maxSB,ketten,nachkommen)

        # Vergleich Zielfunktion
        for i in range(ketten):
            for j in range(nachkommen):
                if zf(xNeu,optimierungsziel,ketten,nachkommen,start)[j,0,i] < zfAlt[0,0,i]:
                    zfverbesserungen = zfverbesserungen + 1
                    zfAlt[0,0,i] = zf(xNeu,optimierungsziel,ketten,nachkommen,start)[j,0,i]
                    zfAlt[0,1,i] = j
                    x = cp.deepcopy(checkNB(xNeu, nb, ketten, nachkommen)[j,:,i])

                elif zf(xNeu,optimierungsziel,ketten,nachkommen,start)[j,0,i] >= zfAlt[0,0,i]:
                    print('Keine Verbesserung der Zielfunktion!')

            if zfverbesserungen == 0:
                maxD = cp.deepcopy(c5 * maxD)
                sbKleinerCounter = sbKleinerCounter + 1
                if sbKleinerCounter >= sbKleiner:
                    print('Zu viele Versuche ohne Verbesserung, Berechnung wird abgebrochen!')
                    exit(1)

            elif zfverbesserungen > 1:
                maxD = cp.deepcopy(c6 * maxD)

            zfverbesserungen = 0


            print(''.join(['ZF-Wert %e, Suchpunkt:' % (zfAlt[0,0,i]), str(x)]))
            zfHistory.append(zfAlt[0,0,i])
            # xHistory[it,:,i] = cp.deepcopy(x)


    #plotResults(zfHistory, xHistory)

run_optimierung(True,50,1,5)
