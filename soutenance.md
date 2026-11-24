# Synthèse

`select_seed`: 
- prend une image le lambda et le padding de l'image
- on pad l'image pour que quand on fait les patchs on ne sorte pas de l'image
- choisi un carré de taille 2lambda * 2lambda dans \[im.width - padding\]\[im.height - padding\]

`get_patches_matrix`:
- prend l'image paddée et la taille des patches
- renvoie une matrice qui contient tous les patchs possible applatis en 1D

`get_patches_score`:
- prend la matrice des patches, un masque binaire qui dit quels pixels ont déjà une valeur et
quel patch on veut check
- on récupère tous les patches de l'image et on leur donne un score par rapport à la distance avec
le patch passé en paramètre

`efros_leung`:
- le padding de l'image est la moitié de la taille d'un patch
- 

`main`:
- 

# Classification

