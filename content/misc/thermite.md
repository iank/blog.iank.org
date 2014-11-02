Title: July 4th with NCSULUG and Thermite
Date: 2014-10-19 17:00
Tags: thermite, america, freedom, chemistry, nerds
Summary: July 4th 2014 Celebration with the [NCSU Linux Users' Group](http://lug.ncsu.edu/), 10kg of [thermite](http://en.wikipedia.org/wiki/Thermite) and a lawnmower. Videos and some grade school chemistry after the break.


<iframe id="ytplayer" type="text/html" width="640" height="390" src="http://www.youtube.com/embed/M9KByjCi3B4?autoplay=0&origin=http://blog.iank.org" frameborder="0" allowfullscreen></iframe>

<p>
</p>

This past July 4th I got together with the [NCSU Linux Users' Group](http://lug.ncsu.edu) and brought about 10kg of [thermite](http://en.wikipedia.org/wiki/Thermite). Above video was expertly filmed and edited by [@coxn](http://twitter.com/coxn). Someone brought a lawnmower. It goes up at about 1:53; we melted straight through the engine block and the rest was in flames almost instantly. Safety precautions were observed.

The thermite reaction, broadly, is an "exothermic reaction which involves a metal reacting with a metallic \[...\] oxide to form a more stable oxide and the corresponding metal \[...\] of the reactant oxide"&nbsp;\[[^wang]\].

A more handwaving and less correct description is that it is a reaction between a metal, usually aluminum, and a metallic oxide, usually iron oxide. The oxygen "switches" between the two, forming aluminum oxide and iron, and producing quite a lot of heat. A typical temperature is on the order of 2,500 kelvins (apx 700 K greater than the melting point of iron).

A general form for the reaction, from \[[^wang]\], is

\begin{equation}
\mathrm{M} + \mathrm{AO} \rightarrow \mathrm{MO} + \mathrm{A} + \Delta H
\end{equation}

where M is some metal or alloy (e.g. Al), and A is a suitable metal or non-metal (e.g. Fe). More exotic thermites have been investigated, \[[^wang]\] has a good overview.

In particular, I made several batches using both "red" iron(III) oxide (i.e., rust) and "black" iron(II,III) oxide. They have slightly different properties, which may also have to do with the powder grain size. In the video the black iron oxide reactions are more violent, throwing off clouds of sparks.

Mixing ratios for various thermites can be found online, and I recomputed them myself as a review of basic chemistry. I've captured the [relevant page from my notebook](/images/thermite_stoichiometry.jpg) and I'll repeat the process for iron(III) oxide here.

Substituting[^ionic] aluminum and iron(III) oxide into the general equation above we have

\begin{equation}
\mathrm{Al} + \mathrm{Fe_2O_3} \rightarrow \mathrm{Al_2O_3} + \mathrm{Fe} + \Delta H
\end{equation}

[Balanced](http://en.wikipedia.org/wiki/Chemical_equation#Balancing_chemical_equations), this becomes

\begin{equation}
2\mathrm{Al} + \mathrm{Fe_2O_3} \rightarrow \mathrm{Al_2O_3} + 2\mathrm{Fe} + \Delta H
\end{equation}

Now work out the [stoichiometric](http://en.wikipedia.org/wiki/Stoichiometry) ratio— convert the equation above, which is in terms of molecules, into an equation in terms of mass. This is done using the [molar](http://en.wikipedia.org/wiki/Mole_(unit)) mass of each term and the result is the ideal ratio of reactants in terms of their mass.

Aluminum has a molar mass of 26.982 g/mol, and iron(III) oxide has 159.687 g/mol (which can be worked out from the molar masses of iron and oxygen using $\mathrm{Fe}_2\mathrm{O}_3$). Taking (from the equation above) 2 mol Al per 1 mol $\mathrm{Fe}_2\mathrm{O}_3$ we have 53.964g Al per 159.687g $\mathrm{Fe}_2\mathrm{O}_3$, or a ratio of **1:2.959**.

Igniting thermite is difficult, requiring a tremendous temperature to get the reaction going. I used magnesium ribbon, which burns at around 3,100 &deg;C. Magnesium ribbon itself is only slightly easier to ignite— it autoignites at 473 &deg;C. I used a propane torch (1,995 &deg;C) to light the magnesium.

Once the reaction starts, there is no good way to stop it as it contains its own oxidizer. Best to do it in a clear area. We kept a fire extinguisher on hand for *after* the reaction in case anything else caught. One of the worst possible ideas is to dump water on it— this will cause a steam explosion, so don't do that. Have some eye protection for the [UVs](http://en.wikipedia.org/wiki/Ultraviolet).

[^wang]: Wang, L_L, Z.A. Munir, and Yu M. Maximov. "Thermite reactions: their utilization in the synthesis and processing of materials." *Journal of Materials Science* 28, no. 14 (1993): 3693-3708.

[^ionic]: It is also necessary to remember about ionic bonds and valence electrons to find the correct form for aluminum oxide.
