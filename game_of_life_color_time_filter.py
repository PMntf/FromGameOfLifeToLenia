import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp
import scipy.signal
#Ajout d'un filtre qui va remplacer la condition des cases adjacentes en ne considérant que la distance avec la case voulue.
#Permets donc de simuler un espace continu

nb_tours = 150

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

mu = 0.7
sigma = 0.3 #valeurs qui peuvent varier et fournir des comportements plus ou moins sensibles

def accroissement(x): #On va rajouter une fonction qui permettra de transformer notre fonction Gaussienne en une fonction  qui servira de taux d'accroissement
    return x*2 - 1

dt =  0.1

R = 10 #rayon du filtre, nombre de cases que l'on veut observer autour du centre
y, x = np.ogrid[-R:R, -R:R] 
distance = np.sqrt((x)**2 + (y)**2) / R #distance des éléments par rapport au centre du filtre
K = transformation(distance, mu, sigma) #On applique notre fonction de transformation pour construire le filtre
# il sera sous la forme d'un anneau grâce à notre fonction de transformation
K = K / np.sum(K) #On normalise finalement le tout

def tour_suivant(plateau): #règle de jeu de la vie
    
    mu = 0.20
    sigma = 0.02
    
    nb_voisins_tab = sp.signal.convolve2d(plateau, K, mode='same', boundary='wrap')
    plateau += dt*accroissement(transformation(nb_voisins_tab, mu, sigma)) 
    plateau = np.clip(plateau, 0, 1) #garde la valeur entre 0 et 1
    
    return plateau

dim = 256

R = dim//4
y, x = np.ogrid[-R:R, -R:R]  #création d'un plateau de base sous la forme d'un disque
distance = np.sqrt((x)**2 + (y)**2) / R 

plateau = np.zeros((dim,dim)) #Génération d'un plateau aléatoire où seulement un carré de (dim//4)*(dim//4) est remplis
pos_x = dim//4
pos_y = dim//4
plateau[pos_x:(pos_x + distance.shape[1]), pos_y:(pos_y + distance.shape[0])] = distance.T

affichage(plateau, tour_suivant, "jeu_de_la_vie.gif", nb_tours, cmap = 'hot')
print("Fin")

