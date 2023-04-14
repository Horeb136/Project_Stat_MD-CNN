# Projet_Stat - MD_CNN

## Pour faire tourner le modèle : 

- [ ] Ouvrir un terminal sur la machine et cloner le dépot git en y exécutant la commande 
```
git clone https://github.com/Horeb136/Project_Stat_MD-CNN.git
```

- [ ] Entrez dans le dossier Project_Stat_MD-CNN en faisant : 
```
cd Project_Stat_MD-CNN
```

- [ ] Le fichier file_to_execute.sh est le fichier qui permet d'installer tous les outils/dépendances nécessaires ainsi que de faire tourner le modèle. Il faut :
    - le rendre exécutable avec la commande 
    ```
    chmod -x file_to_execute.sh
    ```
    - puis l'exécuter avec la commande 
    ```
    ./file_to_execute.sh
    ```

## A la fin de l'exécution du code :

- [ ] Après l'exécution, il faudra nous permettre d'avoir accès au résultat faisant un push. Pour cela, il faut 
    - d'abord retourner dans le dossier Project_Stat_MD-CNN en exécutant : 
    ```
    cd ../../ 
    ```
    - puis exécuter les commandes suivantes : 
    ```
    git add .
    git commit -m "Résultats du modèle md-cnn"
    git push
    ```
