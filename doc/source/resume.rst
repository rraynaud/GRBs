Résumé
******


Pour produire des courbes de lumière associées à la coalescence de
deux étoiles à neutrons, le code GRBs  utilise un modèle adapté des
travaux de `Sun et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017xru..conf..216S/abstract>`_
qui suppose que l'objet compact formé est
un magnétar qui perd de l'énergie par un phénomène de freinage
magnétique induit par la composante dipolaire du champ magnétique de
l'étoile. L'évolution temporelle de la fréquence de rotation de
l'étoile est déterminée par le couple magnétique du rayonnement du
dipôle

.. math::
   \Omega(t) = \frac{\Omega_0}
   {\left(1+t/\tau_{\rm em}\right)^{1/2}},

où :math:`\tau_{\rm em} = 3c^3 I/(B^2 R^6 \Omega_0^2)` est le temps
caractéristique de ralentissement, :math:`I`, :math:`R` et :math:`B` étant le moment
d'inertie, le rayon et le champ magnétique dipolaire de l'étoile à
neutrons. On suppose que le rayonnement X est produit par dissipation
interne dans le vent du magnétar et émis de façon isotrope
`Zhang et al. (2013) <https://ui.adsabs.harvard.edu/abs/2001ApJ...552L..35Z/abstract>`_.
La luminosité en rayons X est alors déterminée par
la luminosité de ralentissement du dipôle, modulo un facteur
d'efficacité :math:`\eta`.

.. math::
  L_X = \eta \Ldip(t) = \eta \frac{B^2 R^6 \Omega^4(t)}{6 c^3}
  \,.


Le magnétar central résultant de la fusion de deux étoiles à neutrons,
il faut également tenir compte de la présence d'éjecta opaques
résiduels qui vont d'abord absorber ce rayonnement dans certaines
directions et le ré-émettre avec un spectre de corps noir. Le calcul
des courbes de lumière dans la zone où le rayonnement est initialement
piégé requiert donc de déterminer l'évolution dynamique des ejecta. On
utilise dans ce cas le modèle de \citet{yu2013}, qui prend en compte
l'injection d'énergie par le magnétar central et le chauffage
additionel par désintégration d'éléments radioactifs.

Enfin, précisons que cette approche phénoménologique permet de
comparer simplement différentes équations d'état qui déterminent le
rayon :math:`R`, le moment d'inertie :math:`I` et la période critique de rotation
du magnétar en deçà de laquelle l'objet s'effondre en trou noir. Cette
période :math:`P` est reliée à la masse maximale d'une étoile à neutrons par
la relation (Lasky 2014)

.. math::
  M_\tx{max} = M_\tx{TOV} (1+ \alpha P^\beta)
  \,,

où la masse maximale d'une étoile statique :math:`M_\text{TOV}` et les
exposants :math:`\alpha` et :math:`\beta` sont donnés par les modèles d'équation
d'état (Ai et al. 2018). Lorsque l'étoile s'effondre en trou noir du
fait du ralentissement de sa rotation, on suppose que la luminosité
due au vent du magnétar s'arrête abruptement.
