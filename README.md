# SINDy-for-Chemical-Kinetics
The system involves 5 species (A-E) and 4 parallel/consecutive reactions with Arrhenius-type kinetics and energy balance.

A → B (k₁)  
A + A → D (k₂)  
A + B → E (k₃)  
B → C (k₄)  

With k_i = k_i0 * exp(-E_i / (8.314 * T))  # Arrhenius equation

dCA_dt = -k1 * CA - k2 * CA^2 - k3 * CA * CB

dCB_dt = k1 * CA - k4 * CB - k3 * CA * CB

dCC_dt = k4 * CB

dCD_dt = k2 * CA^2

dCE_dt = k3 * CA * CB

dT_dt = -(dH1 * k1 * CA + dH2 * k2 * CA^2 + dH3 * k3 * CA * CB + dH4 * k4 * CB) / (cp * p) #batch reactor
 

