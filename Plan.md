c**Mise en contexte**
Les systèmes microélectromécaniques (MEMS) sont partout. Votre téléphone contient un accéléromètre qui détecte l'orientation de l'écran. Le Groupe Aérospatial de l'Université Laval (GAUL) instrumente ses fusées avec des accéléromètres pour enregistrer le profil d'accélération pendant le vol. Ces capteurs reposent sur un principe que vous connaissez bien : le condensateur. Dans ce projet, vous analyserez un accéléromètre MEMS capacitif en utilisant les concepts d'électrostatique vus en classe, puis vous le concevrez pour une application en conditions extrêmes.

**L'accéléromètre capacitif à peignes interdigités**
L'accéléromètre capacitif à peignes interdigités Inspiré de l'ADXL345 d'Analog Devices. Des peignes de plaques conductrices sont imbriqués. Quand le dispositif subit une accélération, une masse d'épreuve (proof mass) se déplace et modifie l'espacement entre les doigts, ce qui change la capacité.

Géométrie simplifiée
• Masse d'épreuve sur ressorts (constante k)
• Doigts conducteurs sur la masse mobile
• Doigts conducteurs fixés au substrat
• Condensateurs plans parallèles

Paramètres typiques
• Épaisseur des doigts : 2 à 5 μm
• Espacement entre doigts : 1 à 4 μm
• Longueur des doigts : 100 à 500 μm
• Nb. de paires de doigts : 20 à 100
• Constante de rappel : 0.1 à 10 N/m
• Masse d'épreuve : ~μg



Petit check avec Claude :)))
$$
\begin{gather}
m\ddot{x}+kx=ma \\
x=\frac{ma}{k} & (1)
\end{gather}
$$

Chaque paire de doigts constitue un condensateur plan

$L$ : longueur
$\ell$ : epaisseur des doigts
$d$: espacement entre les doigts

$$
\begin{gather}
C_{1,\text{ paire}}=\epsilon_{0} \frac{A}{d}=\frac{\epsilon_{0}L \ell}{d} \\
C_{n}=\frac{N\epsilon_{0}L \ell}{d} & (2), \quad \text{(jsp trop pq cest N paires)}
\end{gather}
$$
Ci-haut cest la mm chose que ca a donne a Dara
Suppose un champ électrique uniforme entre les plaques et nul à l'extérieur. Valide lorsque $t\gg d$.

La détection peut être en fonction de la variation de l'espacement entre les doigts ou en fonction de la variation de la surface de recouvrement, *gap closing* vs *overlap*

| *Gap closing* $d=d_{0}\pm x$             | *Overlap* $L=L_{0}\pm x$                  |
| ---------------------------------------- | ----------------------------------------- |
| Réponse non linéaire, sensibilité élevée | Réponse linéaire, sensibilité plus faible |
L'ADXL345 utilise le mode *gap-closing* faudrait de la documentation pour appuyer ca (ca doit bien se trouver)

En mode gap-closing (on fait un coter pour voir que cest pas lineraire et jusftifier dutiliser la SdT)
$$
\begin{gather}
C(a)= \frac{\epsilon NL \ell}{d_{0}-x}  \quad \left( d=d_{0}\pm x\text{ dans (2)} \right)  \\
C(a)=\frac{\epsilon NL \ell}{d_{0}-ma/k}  \quad \text{ subt. de (1)} \\
C(a)=\frac{C_{0}}{1- \frac{ma}{kd_{0}}}
\end{gather}
$$
On peut constater que c'est pas linéaire !!
$$
\begin{gather}
\Delta C=C_{1}-C_{2}=\epsilon_{0}NL \ell\left( \frac{1}{d_{0}-x}-\frac{1}{d_{0}+x} \right) \text{ on revient un peu en arriere} \\
\Delta C=\frac{2\epsilon_{0}NL \ell x}{d_{0}^{2}-x^{2}}=\frac{2C_{0}(ma/k)/d_{0}}{1-(ma/kd_{0})^{2}}
\end{gather}
$$
Wow c'est vraiment pas lineraire non plus !!

linéarisation avec SdT  pour que ce soit pas la mort mais ca rend moins precis

$$
\Delta C \approx \frac{2C_{0}}{d_{0}}\cdot \frac{m}{k}\cdot a\quad x\ll d_{0}
$$
Jai pas checke ca cest le resultat de Claude

Ca donne un sensibilite 
$$
\frac{dC}{da}: \, \frac{d}{da}\left( \Delta C \right) =\frac{2C_{0}m}{kd_{0}}
$$
On considere $\epsilon_{r}=1$ pour lEQ ci-haut, mais dans la vraie vie ya des trucs genre une couche doxydation qui ferait que cest pas vrai

page 16 de ubc_2008_fall_kannan_akila.pdf.pdf, pull in au 2/3 de la distance
page 22 figure + EDO
page 26 graph


$$
\Delta C=\frac{2C_{0}(ma/k)/d_{0}}{1-(ma/kd_{0})^{2}}
$$
$$
\frac{d(\Delta C)}{da}=\frac{4 \epsilon_0 NAm}{kd_0^2}
$$

$$
C=\frac{\epsilon_0 A}{d_0-ma/k}
$$