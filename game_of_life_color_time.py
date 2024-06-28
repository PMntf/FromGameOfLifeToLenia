import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Ajout d'une continuité en temps en changeant la fonction de transformation en un taux d'accroissement. 

nb_tours = 200
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

mu = 3
sigma = 0.5 #valeurs qui peuvent varier et fournir des comportements plus ou moins sensibles

def accroissement(x): #On va rajouter une fonction qui permettra de transformer notre fonction Gaussienne en une fonction  qui servira de taux d'accroissement
    return x*2 - 1

dt =  0.1

def tour_suivant(plateau): #règle de jeu de la vie
    
    nb_voisins_tab = sum(np.roll(np.roll(plateau, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))
    plateau += dt*accroissement(transformation(nb_voisins_tab, mu, sigma)) 
    plateau = np.clip(plateau, 0, 1) #garde la valeur entre 0 et 1
    
    return plateau


remplis = np.random.rand(32, 32) 
plateau = np.zeros((64,64)) #Génération d'un plateau aléatoire où seulement un carré de 32*32 est remplis
pos_x = 64//4
pos_y = 64//4
plateau[pos_x:(pos_x + remplis.shape[1]), pos_y:(pos_y + remplis.shape[0])] = remplis.T

affichage(plateau, tour_suivant, "jeu_de_la_vie.gif", nb_tours, cmap = 'hot')
print("Fin")
