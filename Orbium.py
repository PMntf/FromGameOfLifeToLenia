import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import scipy.signal
#Ajout d'un filtre qui va remplacer la condition des cases adjacentes en ne considérant que la distance avec la case voulue.
#Permets donc de simuler un espace continu

nb_tours = 150
dim = 256

def affichage(plateau, tour_suivant, fichier, nb_etapes, cmap):

    fig = plt.figure(figsize=(12, 12))
    monde = plt.imshow(plateau, cmap=cmap)
    plt.axis('off')
    
    def suiv(i):
        if (i==0):
            return monde
        if (i%10 == 0):
            print(f"{i/10}/{nb_tours/10}")
        nonlocal plateau
        plateau = tour_suivant(plateau)
        monde.set_array(plateau)
        return monde
    
    ani = animation.FuncAnimation(fig, suiv, nb_etapes, interval = 100, cache_frame_data= False)
    ani.save(fichier, fps=30)

def transformation(x, mu, sigma): #on passe a une fonction de filtrage qui va donner la valeur de la case au tour suivant
    return np.exp(-0.5*((x-mu)/sigma)**2)


def accroissement(x): #On va rajouter une fonction qui permettra de transformer notre fonction Gaussienne en une fonction  qui servira de taux d'accroissement
    return x*2 - 1

dt =  0.1

#Filtre
R = 13
y, x = np.ogrid[-R:R, -R:R]
distance = np.sqrt((1+x)**2 + (1+y)**2) / R
mu = 0.5
sigma = 0.15
K = transformation(distance, mu, sigma)
K[distance > 1] = 0
K = K / np.sum(K)     

def tour_suivant(plateau): #règle de jeu de la vie
    
    mu = 0.15
    sigma = 0.015
    
    nb_voisins_tab = sp.signal.convolve2d(plateau, K, mode='same', boundary='wrap')
    plateau += dt*accroissement(transformation(nb_voisins_tab, mu, sigma)) 
    plateau = np.clip(plateau, 0, 1) #garde la valeur entre 0 et 1
    
    return plateau

#structure "Orbium"
orbium = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
plateau = np.zeros((dim,dim)) #Génération d'un plateau aléatoire où seulement un carré de (dim//4)*(dim//4) est remplis
pos_x = dim//4
pos_y = dim//4
plateau[pos_x:(pos_x + orbium.shape[1]), pos_y:(pos_y + orbium.shape[0])] = orbium.T

affichage(plateau, tour_suivant, "jeu_de_la_vie.gif", nb_tours, cmap = 'hot')
print("Fin")


