# Recherche locale qui trouve les meilleures assignations de four à livrer au timestep.
# 0) Solution initiale
# Solution pipo
# 1) Évaluation :
# a) Contraintes molles : bornes des fours, capacité globale, capacité individuelle des camions (passé et futur)
# b) Score : stockage + déplacement (On évalue la qualité d'une solution en faisant une autre recherche locale sur les routes des camions)
# 2) Voisinage :
# tod

# Format : Rendre le voisinage facile à faire, faire en sorte que les conflits soient facilement calculables
# Format : cf. ce qui a déjà été fait
