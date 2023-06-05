import random
import copy as cp
import numpy as np                          # Paket für numerische Operationen
import matplotlib.pyplot as plt             # Paket fürs grafische Darstellen
from matplotlib import rc
import math

zaehler = 0

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
    global zaehler
    zaehler = zaehler + 1
    zSpeicher = np.zeros(shape=(ketten, 1, nachkommen))

    if start:
        for i in range(ketten):
            z = (math.sin(xVec[i, 0, 0]) + 7 * (math.sin(xVec[i, 1, 0])) ** 2 + 0.1 * (xVec[i,2,0]) ** 4 * math.sin(xVec[i, 0, 0]))
            if optimierungsziel:
                zSpeicher[i, 0, 0] = z
            else:
                zSpeicher[i, 0, 0] = z * (-1)

    else:
        for i in range(ketten):
            for j in range(nachkommen):
                z = (math.sin(xVec[i,0,j]) + 7 * (math.sin(xVec[i,1,j]))**2 + 0.1 * (xVec[i,2,j])**4 * math.sin(xVec[i,0,j]))
                if optimierungsziel:
                    zSpeicher[i,0,j] = z
                else:
                    zSpeicher[i,0,j] = z*(-1)

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

    maxSB = np.zeros(shape=(ketten,3,2))
    minSB = np.zeros(shape=(ketten,3,2))
    for i in range(ketten):
        maxSB[i, :, 0] = x[i,:,0] - (0.5 * maxD * c1)
        maxSB[i, :, 1] = x[i,:,0] + (0.5 * maxD * c1)
        minSB[i, :, 0] = x[i,:,0] - (0.5 * maxD * c2)
        minSB[i, :, 1] = x[i,:,0] + (0.5 * maxD * c2)
        # print('maxSB: ')
        # print(str(maxSB))

    return maxSB, minSB


def getNachkomme(minSB, maxSB,ketten,nachkommen):
    """
    Generiere Nachkomme innerhalb vordefiniertem Suchbereich

    :param minSB:
    :param maxSB:
    :return: Koordinaten Nachkomme (im Eingangsraum)
    """

    # Vektoren (für 3 dimensionalen Eingangsraum!)
    xNachkomme = np.zeros(shape=(ketten, 3, nachkommen))

    for i in range(ketten):
        for j in range(nachkommen):
            xNachkomme[i, 0, j] = random.uniform(maxSB[i, 0, 0], minSB[i, 0, 0])
            xNachkomme[i, 1, j] = random.uniform(minSB[i, 1, 0], maxSB[i, 1, 0])
            xNachkomme[i, 2, j] = random.uniform(minSB[i, 2, 0], maxSB[i, 2, 0])

    # print('getNachkomme: ')
    # print(str(xNachkomme))
    return xNachkomme


def checkNB(xVec, nb):
    """
    Prüfung der Nebenbedingungen, bei Verletzung der Nebenbedingung wird der Punkt auf NB zurückgeschoben

    :param xVec: Vektor im Eingangsraum
    :param nb: Nebenbedingung
    :return: (korrigierter) Vektor im Eingangsraum
    """
    # print('xVec NB: ')
    # print(str(xVec))

    for k in range(3):
        if xVec[k] < nb[k, 0]:
            xVec[k] = cp.deepcopy(nb[k, 0])
        if xVec[k] > nb[k, 1]:
            xVec[k] = cp.deepcopy(nb[k, 1])

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
# optimierungsziel ist True(min) oder False(max -> zf * (-1))
# Anzahl an Iterationen
# Ketten Anzahl (wieviele Startpunkte)
# Nachkommen = Wieviel Kinder je Kette (vielfaches von 5), nur mit dem besten Weiter
def run_optimierung(optimierungsziel,iterationen,ketten,nachkommen):

    if nachkommen % 5 > 0:
        print('Die Anzahl der Nachkommen muss ein Vielfaches von 5 sein!')
        exit(1)

    zfAlt = np.ndarray(shape=(ketten,2,1))

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
    sbKleiner = 20*ketten
    sbKleinerCounter = 0
    zfverbesserungen = 0


    # Initiale Startpunkte
    start = True
    x = np.ndarray(shape=(ketten, 3, 1))
    # print('start x: ')
    # print(str(x))
    for i in range(ketten):
        x[i,0,0] = random.uniform(nb[0, 0], nb[0, 1])
        x[i,1,0] = random.uniform(nb[1, 0], nb[1, 1])
        x[i,2,0] = random.uniform(nb[2, 0], nb[2, 1])
    # print('nach random x: ')
    # print(str(x))

    # Optimierungsschleife
    zfHistory = []
    xHistory = np.zeros(shape=(ketten, 3, iterationen))

    for it in range(iterationen):
        print('Iteration: %i' %it)
        # Schrittweiten/Suchbereichsdefinition
        maxSB, minSB = suchBereichBestimmen(x, maxD, c1, c2, ketten)
        # Zielfunktionsauswertung
        if start:
            for i in range(ketten):
                zfAlt[i,0,0] = zf(x,optimierungsziel,ketten,nachkommen,start)[i,0,0]
                zfAlt[i,1,0] = 0

        start = False
        # Erzeugen eines Nachkommens
        xNeu = getNachkomme(minSB,maxSB,ketten,nachkommen)

        # Vergleich Zielfunktion
        for i in range(ketten):
            for j in range(nachkommen):
                if zf(xNeu,optimierungsziel,ketten,nachkommen,start)[i,0,j] < zfAlt[i,0,0]:
                    zfverbesserungen = zfverbesserungen + 1
                    zfAlt[i,0,0] = zf(xNeu,optimierungsziel,ketten,nachkommen,start)[i,0,j]
                    # print('step 1: ' + str(zf(xNeu,optimierungsziel,ketten,nachkommen,start)[i,0,j]) + ' j: ' + str(j) + ' i: ' + str(i))
                    zfAlt[i,1,0] = j
                    # print('zfAlt: ')
                    # print(str(zfAlt))
                    # print('x: ')
                    # print(str(x))
                    # print('x_neu: ')
                    # print(str(xNeu))
                    x[i, 0, 0] = xNeu[i, 0, j]
                    x[i, 1, 0] = xNeu[i, 1, j]
                    x[i, 2, 0] = xNeu[i, 2, j]


                elif zf(xNeu,optimierungsziel,ketten,nachkommen,start)[i,0,j] >= zfAlt[i,0,0]:
                    print('Keine Verbesserung der Zielfunktion!')

            if zfverbesserungen == 0:
                maxD = cp.deepcopy(c5 * maxD)
                sbKleinerCounter = sbKleinerCounter + 1
                if sbKleinerCounter >= sbKleiner:
                    print('Zu viele Versuche ohne Verbesserung, Berechnung wird abgebrochen!')
                    exit(1)

            elif zfverbesserungen > 1:
                maxD = cp.deepcopy(c6 * maxD)
            # print('maxD: ')
            # print(str(maxD))

            zfverbesserungen = 0

            print('Zielfunktionsaufrufe: '+ str(zaehler))
            print(''.join(['ZF-Wert: %e' % (zfAlt[i,0,0])]))
            print('Suchpunkt: ')
            print(str(x))

            zfHistory.append(zfAlt[i,0,0])
            # xHistory[it,:,i] = cp.deepcopy(x)


    #plotResults(zfHistory, xHistory)

run_optimierung(False,50,3,5)
