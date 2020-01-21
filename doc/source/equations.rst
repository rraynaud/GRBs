Modelling
*********

Governing equations
===================

We consider a magnetar of radius :math:`R`, mass :math:`M`, moment of inertia :math:`I`
and surface magnetic field :math:`B`, surrounded by an accretion disc with
initial mass :math:`M_\tx{disc}` and radius :math:`R_\tx{disc}`. The time
evolution of the magnetar angular frequency :math:`\Omega` will be
determined both by the magnetic and accretion torques :math:`\Ndip` and
:math:`\Nacc`, defined respectively by

.. math::
   \begin{align}
   \Ndip &= - \frac{\mu^2 \Omega^3}{6 c^3} \,,\\
   \Nacc &= n(\Omega) \left(G M \alfvenradius \right)^{1/2} \dot{M}
   \,,
   \end{align}
   :label: e:torques

where :math:`\mu = B R^3` is the dipole moment and :math:`\alfvenradius` the
Alfvén radius

.. math::
   \begin{equation}
   \alfvenradius = \mu^{4/7} \left(GM\right)^{-1/7} \dot{M}^{-2/7}
   \,.
   \end{equation}

Depending of the values of the Alfvén radius and the corotation
radius :math:`\corotationradius = \left(GM/\Omega^2\right)^{1/3}`, the
system will be accreting or expelling material:

- if :math:`\alfvenradius <\corotationradius`, the system is accreting and the accretion torque
  spins up the magnetar,

- if :math:`\alfvenradius > \corotationradius`, the magnetar loses
  angular momentum with the expelled material in the so-called
  propeller regime Gompertz et al. 2014).

The change of sign of the accretion torque is handle by the following prefactor

.. math::
   \begin{equation}
   n(\Omega) =
   \begin{cases}
   \begin{aligned}
   &1 - \left(\frac{\alfvenradius}{\corotationradius}\right)^{3/2}  &\alfvenradius> R\\
   &1 - \frac{\Omega}{\Omega_\tx{K}}  &\alfvenradius < R
   \end{aligned}
   \end{cases}
   \,,
   \end{equation}

where :math:`\Omega_\tx{K}=\sqrt{GM/R^3}`.

TO DO

* BAR MODE INSTA
* accretion rate
* ejecta

The complete ODE system to be integrated in time is then
(Sun et al. 2017)

.. math::
   \begin{align}
   \dot{\Omega} &= \frac{\Ndip + \Nacc}{I} \,,\\
   \dot{t}'     &= \Doppler(\Gamma) \,,\\
   \dot{\Gamma} &= \frac{(\Ldip + \Lrad - \Lelec)-
   \tfrac{\Gamma}{\Doppler}(\xi \Ldip + \Lrad - \Lelec)+
   \Gamma\Doppler\tfrac{\Eint'}{3V'}(4\pi R^2\beta c)}
   {M_\tx{ej}c^2+ \Eint'} \,,\\
   \dot{E}_\tx{int}' &= \Doppler\left[\tfrac{1}{\Doppler^2}(\xi \Ldip + \Lrad - \Lelec)-
   \tfrac{\Eint'}{3V'}(4\pi R^2\beta c)\right] \,,\\
   \dot{V}' &= 4\pi R^2\beta c \Doppler \,,\\
   \dot{R} &= \frac{\beta c}{1-\beta}
   \,.
   \end{align}

In the above system, :math:`\Doppler` is the Doppler factor defined by
:math:`\Doppler = [\Gamma(1-\beta\cos\theta)]^{-1}` with :math:`\beta(\Gamma) =
(1-\Gamma^{-2})^{-1/2}` and :math:`\xi` an efficiency parameter defining the
fraction of the spin-down energy that is used to heat the ejecta. The
different luminosities that enter the equations are the dipole
spin-down luminosity :math:`\Ldip`, the co-moving radiative heating
luminosity :math:`\Lrad` and the co-moving bolometric emission luminosity of
the heated electrons :math:`\Lelec`, respectively.

.. note:: The above system is build in :py:meth:`grb.GRB.Build_RHS`
          thas is passed to the :py:meth:`grb.GRB.Time_integration` routine.

.. math::
   \begin{align}
   \Ldip &= \frac{B^2 R^6 \Omega(t)^4}{6c^3} \,,\\
   \Lrad &=  4 \times 10^{49} M_\tx{ej,-2}
   \left[\frac{1}{2}-\frac{1}{\pi} \arctan
   \left(\frac{t'-{1.3}\,\mathrm{s}}{{0.11}\,\mathrm{ s}}\right)
   \right]^{1.3} \Doppler^2 \,,\\
   \Lelec' &=
   \begin{cases}
   \begin{aligned}
   & c\Eint' \tfrac{\Gamma}{\tau R} &t < t_{\tau=1} \\[1ex]
   & c\Eint' \tfrac{\Gamma}{R} &t \geq t_{\tau=1}
   \end{aligned}
   \end{cases}
   \,.
   \end{align}


Optical depth of the ejecta

.. math::
   \begin{equation}
   \tau = \kappa \frac{M_\tx{ej}}{V'}\frac{R}{\Gamma}
   \end{equation}


Blackbody spectrum

.. math::
   \begin{align}
   T' &=
   \begin{cases}
   \begin{aligned}
   &\left(\frac{\Eint'}{a V' \tau}\right)^{1/4} & \tau > 1 \\
   &\left(\frac{\Eint'}{a V'}\right)^{1/4} & \tau \leq 1
   \end{aligned}
   \end{cases} \,,\\
   \nu \Lbb &= \frac{8 \pi^2 \Doppler^2 R^2}{h^3 c^2}
   \frac{(h \nu/\Doppler)^4}{\exp\left(h\nu/\Doppler kT'\right)-1}
   \,.
   \end{align}

where :math:`a` and :math:`k` are respectively the blackbody radiation constant
and the Boltzmann constant (in CGS units, we have
:math:`a=7.5646\times 10^{-15}\,\mathrm{erg\, cm^{-3}\,K^{-4}}` and
:math:`k=1.380658\times10^{-16}\,\mathrm{erg/K}`).

Code outputs
============

The main output of the code are the X-ray luminosities for both the
free and trapped zones

.. math::
   \begin{align}
   L_\tx{X,free}(t) &= \eta \Ldip(t) \,,\\
   L_\tx{X,trapped}(t) &= e^{-\tau} \eta \Ldip(t) + \int_{{0.3}\,\mathrm{keV}}^{{6}\,\mathrm{keV}} \nu \Lbb d \nu
   \,.
   \end{align}


where one needs to introduce the factor :math:`\eta` to parametrize the
efficiency at which the dipole spin-down luminosity is converted to
X-ray luminosity.


magnetar collapse

.. note:: The code has been benchmark against the results of Gompertz et al. (2014).

Luminosities:

.. math::
   \begin{eqnarray}
   L_\tx{e}		&=& \mathcal{D}^2\frac{E_\tx{int}'c}{R/\Gamma}\times
   \left\{\begin{array}{ll}
   \tau^{-1}&\quad\tx{for }t<t_\tau, \\
   1 &\quad\tx{for }t\geq t_\tau,  \end{array}\right. \\
   L_\tx{ra}		&=& 4\times10^{49}M_\tx{ej,-2}\times
   \mathcal{D}^2\left[\frac{1}{2}-\frac{1}{\pi}\arctan\left(\frac{t'-t}{t'_\sigma}\right)\right] \\
   L_\tx{sd}		&=& L_\tx{dip}+L_\tx{prop} \\
   L_\tx{dip}		&=& -\eta_\tx{dip} N_\tx{dip} \Omega \\
   L_\tx{prop}	&=& -\eta_\tx{prop}\left[ N_\tx{acc} \Omega +\frac{GM_*\dot{M}}{r_m}\right]
   \end{eqnarray}

Torques:

.. math::
   \begin{eqnarray}
   N_\tx{dip}		&=& -\frac{B^2R_*^6\Omega^3}{6c^3} \\
   N_\tx{acc}		&=& \dot{M}\sqrt{GM_*R_*}
   \left\{\begin{array}{ll}
   \left(1-\left(\tfrac{r_m}{r_\tx{c}}\right)^{3/2}\right)&\quad\tx{for } r_m>R_*, \\
   \left(1-\tfrac{\Omega}{\Omega_\tx{K}}\right)&\quad\tx{for } r_m<R_*,
   \end{array}\right. \\
   N_\tx{gw}		&=& -\frac{32GI^2\epsilon^2\Omega^5}{5c^5}
   \end{eqnarray}

Other functions:

.. math::
   \begin{eqnarray}
   \dot{M}		&=& \frac{M_\tx{disk}}{\tau_\alpha}e^{-t/\tau_\alpha} \\
   \tau_\alpha	&=& \frac{R_\tx{disk}^2}{3\alpha c_s H} \\
   \tau			&=& \kappa\frac{M_\tx{ej}}{V'}\frac{R}{\Gamma} \\
   c_s 			&=& H\Omega_\tx{K}\left(\frac{R_*}{R_\tx{disk}}\right)^{3/2}
   \end{eqnarray}


Postprocessing
==============

You can use the methods :py:meth:`grb.GRB.PlotLuminosity` and :py:meth:`grb.GRB.PlotRadii` to visualise the results and :py:meth:`grb.GRB.WriteTable` to store them in your preferred format.
