import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def affichage(plateau, tour_suivant, fichier, nb_etapes, cmap):

    fig = plt.figure(figsize=(16, 16))
    monde = plt.imshow(plateau, cmap=cmap)
    plt.axis('off')
    
    def suiv(i):
        if (i==0):
            return monde
        nonlocal plateau
        plateau = tour_suivant(plateau)
        monde.set_array(plateau)
        return monde
    
    ani = animation.FuncAnimation(fig, suiv, nb_etapes, interval = 100, cache_frame_data= False)
    ani.save(fichier, fps=30)

def tour_suivant(plateau): #règle de jeu de la vie
    nb_voisins_tab = sum(np.roll(np.roll(plateau, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))  
    return (nb_voisins_tab == 3) | (plateau & (nb_voisins_tab == 2))


plateau = np.random.randint(0, 8, (256, 256)) #Génération d'un plateau aléatoire 
plateau = (plateau==0)
nb_tours = 100

affichage(plateau, tour_suivant, "jeu_de_la_vie.gif", nb_tours, cmap = 'Greys')