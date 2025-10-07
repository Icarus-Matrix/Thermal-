%% This script will be used to solve the energy balance for the fuselage and avionics of the glider.

%The script will solve an energy balance for each avionics componenet and the fuselage based on the following heat transfer paths   
    %Solar Irradiation from the sun 
    %Albedo and irradiance from the earth 
    %radiosity network for internal fuselage network
    %radiosity of fuselage to surroundings
    %external convection (while ignore natural convection


%% main function to find Temperatures of Avionics and Fuselage%% main function to find Temperatures of Avionics and Fuselage
function [internalHTC,airTemperature,temperatureTrackArray, qnetArray, outCondArray, convectionArray, ...
          absSolarIrradianceArray, radiosityArray, insideConductionArray, ...
          timeArray, payloadTemp, payloadNetPower, powerbrickTemp, powerbrickNetPower, ...
          auterionTemp, auterionNetPower, gpsTemp, gpsNetPower, ethernetTemp, ethernetNetPower, ...
          radioTemp, radioNetPower, battery1Temp, battery1NetPower, battery2Temp, battery2NetPower, ...
          battery3Temp, battery3NetPower, battery4Temp, battery4NetPower, ...
          dTpayload, dTpowerbrick, dTauterion, dTgps, dTethernet, ...
          dTradio, dTbattery1, dTbattery2, dTbattery3, dTbattery4, ...
          dTwallinside, dTwalloutside,TwallinsideTemp, TwallinsideNetPower, TwalloutsideTemp, Qwallout,silvusTemp,silvusNetPower] ...
          = thermalAnalysis(altitude, latitude, day, solarTime, airSpeed, totalTime,vent_velocity)

%% initalizaiton 
    
    % worksheet 
    avionics=readtable("C:\Users\BrianByrne\Desktop\Thermal Analysis\VIEWFACTOR GEOMETRY.xlsx","Sheet","Avionics Geometry"); % Excel units: qgen [W/m^2 per surface], areas [m^2], k [W/m-K], thickness [m], rho [kg/m^3], Cp [J/kg-K]
    
    % constants
    P0=101325; %Pa at sea level 
    Rair=287.05; %J/kgK at sea level 
    sigma=5.67e-8; 
    
    %fuselage geometry 
    fuselageLength = 1154*.001;    % m
    fuselageSurfaceArea = 2.486; %m^2
    fuselageMass=1150*.002; %kg
    
    %%heat sink parameters 
    heatsinkthickness=.0015; %M
    heatsinkK=240;

    % parameters from excel sheet 
    Nsurfaces=size(avionics,1); %defines number of surfaces from excel sheet 
    density=avionics{:,11}; %kg/m^3
    specificHeats=avionics{:,12}; %J/kgK
    emissivities=avionics{:,15};
    absorptivities=avionics{:,16};
    thickness=avionics{:,18}; %effective thickness is used to solve energy balances for surfaces found by Volume/Area to properly weight thermal mass 
    k=avionics{:,13}; %W/mk
    areas=avionics{:,10}; %m^2
    qgen=avionics{:,17};
    avionicTemperatures=avionics{:,14}; %initial temperatures 
    Twalloutside=avionicTemperatures(Nsurfaces);
    Twallinside=Twalloutside;

    % time stepping parameters
    delta = 0.01;                 % timestep [s]
    storeEverySec = 10;           % how often to store [s]
    storeEverySteps = round(storeEverySec/delta);

    nsteps = floor(totalTime/delta) + 1;                 % include t=0
    nstore = floor((nsteps-1)/storeEverySteps) + 1;      % number of stored points

    % surface groups 
    r2sPayload   = [1 2 3 4 5 6];
    r2sPowerbrick= [7 8 9 10 11 12];
    r2sAuterion  = [13 14 15 16 17 18];
    r2sGPS       = 19;
    r2sethernet  = 20;
    r2sRadio     = [21 22 23 24 25 26];
    r2sBattery1  = [27 28 29 30 31 32];
    r2sBattery2 = [33 34 35 36 37 38];
    r2sBattery3 = [39 40 41 42 43 44];
    r2sBattery4 = [45 46 47 48 49 50];
    r2sSilvus= [51 52 53 54 55 56]; 

%% preallocation 
    %general temperature and time 
    temperatureTrackArray = zeros(Nsurfaces+1,nstore); %%used to store temperatures for each surface during each time step 
    timeArray             = zeros(nstore,1); %%used to store time step 
    
    %fuselage energy balance components
    qnetArray             = zeros(Nsurfaces,nstore); %% net heat (W/m^2) in/out of the fuselage 
    outCondArray          = zeros(nstore,1); 
    convectionArray       = zeros(nstore,1);
    absSolarIrradianceArray=zeros(nstore,1);
    radiosityArray        = zeros(nstore,1);
    insideConductionArray = zeros(nstore,1);
    TwalloutsideTemp      = zeros(nstore,1);
    Qwallout              = zeros(nstore,1);
    % preallocate heat fluxes and temperatures arrays 
    %payload 
    payloadTemp           = zeros(nstore,1);
    payloadNetPower       = zeros(nstore,1);  
    %powerbrick 
    powerbrickTemp        = zeros(nstore,1);
    powerbrickNetPower     = zeros(nstore,1);
    %auterion 
    auterionTemp          = zeros(nstore,1);
    auterionNetPower       = zeros(nstore,1);
    %gps
    gpsTemp               = zeros(nstore,1);
    gpsNetPower            = zeros(nstore,1);
    %ethernet
    ethernetTemp          = zeros(nstore,1);
    ethernetNetPower       = zeros(nstore,1);
    %radio
    radioTemp             = zeros(nstore,1);
    radioNetPower          = zeros(nstore,1);
    %battery 1
    battery1Temp          = zeros(nstore,1);
    battery1NetPower       = zeros(nstore,1);
    %battery 2 
    battery2Temp          = zeros(nstore,1);
    battery2NetPower        = zeros(nstore,1);
    %battery 3
    battery3Temp          = zeros(nstore,1);
    battery3NetPower        = zeros(nstore,1);
    %battery 4
    battery4Temp          = zeros(nstore,1);
    battery4NetPower       = zeros(nstore,1);
    %inside fuselage wall 
    TwallinsideTemp       = zeros(nstore,1);
    TwallinsideNetPower   = zeros(nstore,1);
    %silvus temperature 
    silvusTemp           = zeros(nstore,1);
    silvusNetPower        = zeros(nstore,1);
    %net heat from radiosity network 
    qnet = zeros(Nsurfaces,1);

    % Preallocate derivatives (dT/dt) for storage
    dTpayload       = zeros(nstore,1);
    dTpowerbrick    = zeros(nstore,1);
    dTauterion      = zeros(nstore,1);
    dTgps           = zeros(nstore,1);
    dTethernet      = zeros(nstore,1);
    dTradio         = zeros(nstore,1);
    dTbattery1      = zeros(nstore,1);
    dTbattery2      = zeros(nstore,1);
    dTbattery3      = zeros(nstore,1);
    dTbattery4      = zeros(nstore,1);
    dTSilvus        = zeros(nstore,1);
    dTwallinside    = zeros(nstore,1);
    dTwalloutside   = zeros(nstore,1);
    dTdt = zeros(Nsurfaces,1);


    % view factors
    Nrays=1000;
    viewfactorMatrix=avionicsViewFactors(Nrays); 

    % time marching
    z = 1;  % storage index

    %%convecting terms
    [airDensity,airTemperature,pressure]=getAirProperties(altitude,Rair); %% air properties
    htc=getHTC(pressure,airTemperature,airDensity,fuselageLength,airSpeed); %outside convection coefficient 
    internalHTC=getinternalHTC(vent_velocity,airDensity,airTemperature,pressure); %%internal convection 
  
   %% explicit solver for energy balances 
  for w = 1:nsteps
     
        %%simulation time 
        tcurrent=(w-1)*delta;

        %%update parameters 
        [solarIrradiance,albedoLoad]=getSolarLoad(day,pressure,P0,solarTime,latitude); %update solar load 
        J=getRadiosity(Nsurfaces,emissivities,viewfactorMatrix,avionicTemperatures,areas); %update radiosity (returns in units of W/m^2)

        %find avionic and fuselage net heat fluxes from radiosity network 
        for i=1:Nsurfaces
            
            %net heat flux from radiosity net work 
            qnet(i) = viewfactorMatrix(i,:) * J - J(i);

  %% energy balances 
     %%fuselage outside wall energy balance: Qsolar+Qalbedo+Qconduction-Qconvection-Qradiosity=Estored
     %%fuselage inside wall energy balance: Qnetradiation-Qconduction=Estored
     %%avionics energy balance: Qnetradiation+Qgen=Estored

            if i==Nsurfaces %runs only for last value in loop (fuselage)

                %% outside wall energy balance parameters (W)
                outsideConduction=fuselageSurfaceArea*(k(i)/thickness(i))*(Twallinside-Twalloutside); %%outside conduction (W)
                convection=fuselageSurfaceArea*htc*(Twalloutside-airTemperature); %%convection 
                absSolarIrradiance=(fuselageSurfaceArea/2)*(absorptivities(i)*(solarIrradiance+albedoLoad)); %%absorbed solar irradiance (assume solar irradiation and albedo only sees half of surface area) (W)
                emittedRadiation=(emissivities(i)*sigma*fuselageSurfaceArea*(Twalloutside^4-airTemperature^4)); %emmited radition from surface  
                Qout=absSolarIrradiance+outsideConduction-convection-emittedRadiation; %W
                Twalloutsidenew=(delta/(fuselageMass*specificHeats(i)))*Qout+Twalloutside; %%update outside wall temperature  (K)
                dTdt_out = Qout / (fuselageMass*specificHeats(i)); %%rate of temperature change [K/s]
                Twalloutside=Twalloutsidenew; %update temperature 
                
                %% inside wall energy balance 
                insideConduction=fuselageSurfaceArea*(k(i)/thickness(i))*(Twallinside-Twalloutside); %(W)
                Twallinsidenew=(delta/(fuselageMass*specificHeats(i)))*((qnet(i)*fuselageSurfaceArea)-insideConduction)+Twallinside; %%update inside wall temperature (K)
                dTdt_in = ((qnet(i)*fuselageSurfaceArea)-insideConduction) / (fuselageMass*specificHeats(i)); %Rate of temperature change [K/s]
                Twallinside=Twallinsidenew; %updated temperature 
                avionicTemperatures(Nsurfaces)=Twallinside;

                %%store value every 10s for output 
                if mod(w-1,storeEverySteps)==0
                    temperatureTrackArray(i,z)=Twallinside;
                    temperatureTrackArray(i+1,z)=Twalloutside;
                    qnetArray(i,z)=qnet(i);
                    outCondArray(z)=outsideConduction;
                    convectionArray(z)=convection;
                    absSolarIrradianceArray(z)=absSolarIrradiance;
                    radiosityArray(z)=emittedRadiation;
                    insideConductionArray(z)=insideConduction; 
                    timeArray(z)= (w*delta)/3600;   
                    Qwallout(z)=Qout;
                end 

            else
                
                %% avionic energy balanace 
                qavionicConvection=internalHTC*(avionicTemperatures(i)-airTemperature); %(W/m^2)
                if i==6 || i==56 %%incorperates payload conduction into heat sink 
                    qConduction=(heatsinkK/heatsinkthickness)*(avionicTemperatures(i)-airTemperature); %%conduction into heat sink 
                    avionicTempNew=(delta/(density(i)*thickness(i)*specificHeats(i)))*(qnet(i)+qgen(i)-qavionicConvection-qConduction)+avionicTemperatures(i); %%update avionic temperature (K)
                else
                avionicTempNew=(delta/(density(i)*thickness(i)*specificHeats(i)))*(qnet(i)+qgen(i)-qavionicConvection)+avionicTemperatures(i); %%update avionic temperature (K) 
                end 
                dTdt(i) = (qnet(i)+qgen(i)-qavionicConvection) / (density(i)*thickness(i)*specificHeats(i)); %%derivative 
                avionicTemperatures(i)=avionicTempNew;
                
                %%store value every 10 seconds
                if mod(w-1,storeEverySteps)==0
                    temperatureTrackArray(i,z)=avionicTemperatures(i);
                    qnetArray(i,z)=qnet(i);
                end 

            end
                
        end %end surface energy balance 
        

  %% solar time 
        solarTime = solarTime + delta/3600;  % advance in hours
        if solarTime >= 24
            solarTime = solarTime - 24;      % wrap daily cycle
        end

 %% store valeus 
        if mod(w-1,storeEverySteps)==0
            %store avionic net powers (W)
            payloadNetPower(z)     = sum(qnet(r2sPayload)   .* areas(r2sPayload));
            powerbrickNetPower(z)  = sum(qnet(r2sPowerbrick).* areas(r2sPowerbrick));
            auterionNetPower(z)    = sum(qnet(r2sAuterion)  .* areas(r2sAuterion));
            gpsNetPower(z)         = sum(qnet(r2sGPS)       .* areas(r2sGPS));
            ethernetNetPower(z)    = sum(qnet(r2sethernet)  .* areas(r2sethernet));
            radioNetPower(z)       = sum(qnet(r2sRadio)     .* areas(r2sRadio));
            battery1NetPower(z)    = sum(qnet(r2sBattery1)  .* areas(r2sBattery1));
            battery2NetPower(z)    = sum(qnet(r2sBattery2)  .* areas(r2sBattery2));
            battery3NetPower(z)    = sum(qnet(r2sBattery3)  .* areas(r2sBattery3));
            battery4NetPower(z)    = sum(qnet(r2sBattery4)  .* areas(r2sBattery4));
            silvusNetPower(z)      = sum(qnet(r2sSilvus)    .* areas(r2sSilvus));
            TwallinsideNetPower(z) = qnet(Nsurfaces)        * areas(Nsurfaces);
            %store derivates
            dTpayload(z)    = mean(dTdt(r2sPayload));
            dTpowerbrick(z) = mean(dTdt(r2sPowerbrick));
            dTauterion(z)   = mean(dTdt(r2sAuterion));
            dTgps(z)        = dTdt(r2sGPS);
            dTethernet(z)   = dTdt(r2sethernet);
            dTradio(z)      = mean(dTdt(r2sRadio));
            dTbattery1(z)   = mean(dTdt(r2sBattery1));
            dTbattery2(z)   = mean(dTdt(r2sBattery2));
            dTbattery3(z)   = mean(dTdt(r2sBattery3));
            dTbattery4(z)   = mean(dTdt(r2sBattery4));
            dTSilvus(z)     = mean(dTdt(r2sSilvus));
            dTwallinside(z) = dTdt_in;
            dTwalloutside(z)= dTdt_out;
            timeArray(z)=tcurrent/3600;
            output="Time computed: " + string(tcurrent);
            disp(output)
            z=z+1;
        end 
  end 

 %% average Componenet Temperatures (K)
    for z=1:nstore
        %%surface sums for heat flux 

        %%surface sums 
        payloadTemp(z)     = mean(temperatureTrackArray(r2sPayload,z));
        powerbrickTemp(z)  = mean(temperatureTrackArray(r2sPowerbrick,z));
        auterionTemp(z)    = mean(temperatureTrackArray(r2sAuterion,z));
        gpsTemp(z)         = mean(temperatureTrackArray(r2sGPS,z));
        ethernetTemp(z)    = mean(temperatureTrackArray(r2sethernet,z));
        radioTemp(z)       = mean(temperatureTrackArray(r2sRadio,z));
        battery1Temp(z)    = mean(temperatureTrackArray(r2sBattery1,z));
        battery2Temp(z)    = mean(temperatureTrackArray(r2sBattery2,z));
        battery3Temp(z)    = mean(temperatureTrackArray(r2sBattery3,z));
        battery4Temp(z)    = mean(temperatureTrackArray(r2sBattery4,z));
        silvusTemp(z)      = mean(temperatureTrackArray(r2sSilvus,z));
        TwallinsideTemp(z) = mean(temperatureTrackArray(Nsurfaces,z));
        TwalloutsideTemp(z)= mean(temperatureTrackArray(Nsurfaces+1,z));
    end %end loop for surface sums 

end %end function 


%stratospheric air property calculator 
function [density,temperature,pressure]=getAirProperties(altitude,Rair)
  
    altitude=altitude*1000; %convert to meters
    tropospherePressure=22632; %Pa
    troposphereTemperature=216.65; %k
    baseHeight=11000; %m
    
    %pressure 
    pressure=tropospherePressure*exp(-1*(9.81*(altitude-baseHeight))/(Rair*troposphereTemperature));

    %temperature
    if altitude<20000
        temperature=216.65; %isothermal conditions
    else 
        temperature=216.65+.001*(altitude-20000); %temperature rises due to increase in solar irradiance
    end 
    
    %density
    density=(pressure)/(Rair*temperature);
end 


%This function calculates the solar irradiation for a given
%time,day,latitude and altitude
function [solarIrradiance,albedoLoad]=getSolarLoad(day,pressure,P0,solarTime,latitude)
    sunIrradiance=1367; %w/m^2
    e=.0167; %earth's orbital eccentricity

    %anomaly calcs
    MA=(2*pi*day)/365; %mean anomaly 
    trueAnomaly=MA+2*e*sin(MA)+1.25*e^2*sin(2*MA);
    
    %angle between surface normal and solar rays clacs
    omega1=deg2rad(15*(12-solarTime)); %angular displacement of sun from local solar noon
    omega2 = deg2rad(23.45 * sin(deg2rad((360/365)*(284+day)))); %solar declination
    omega3=latitude*(pi/180); %latitude in radians 
    zenith=sin(omega3)*sin(omega2)+cos(omega2)*cos(omega1)*cos(omega3);
    
   
    M=(pressure/P0)*(sqrt(1229+(614*zenith)^2)-614*zenith); %air mass ratio 
    transmittance=.5*(exp(-65*M)+exp(-.95*M)); %transmittance at stratophere 

    %solar irradiance
    solarIrradiance=sunIrradiance*((1+e*cos(trueAnomaly))/(1-e^2))^2*transmittance^M;

    %albedo 
    albedoLoad=.3*solarIrradiance;
end 

function htc=getHTC(pressure,temperature,density,fuselageLength,airSpeed)
    %Nusselt correlations for flow over isothermal flat plate will be used.  %%surface sums 
    %Air properties will be extracted via CoolProp

    %thermo properties of air at pressure and temperature 
    mu=double(py.CoolProp.CoolProp.PropsSI("VISCOSITY", "T", temperature, "P", pressure, "Air")); %Pa-s (dynamic viscosity)
    k=double(py.CoolProp.CoolProp.PropsSI("CONDUCTIVITY", "T", temperature, "P", pressure, "Air")); %W/mK (thermal conductivity)
    Cp=double(py.CoolProp.CoolProp.PropsSI("Cpmass", "T", temperature, "P", pressure, "Air")); %J/kgK (specific heat)
    
    %%reynolds and Prandtl number calculations 
    Re=(density*airSpeed*fuselageLength)/mu;
    Pr=(Cp*mu)/k;

    %%Nusselt Correlation calculation
    if Re<5*10^5
        Nu=.664*Re^(1/2)*Pr^(1/3);
    elseif Re>=5*10^5 && Re<3.8*10^6
        Nu=(0.03*Re^.8-871)*Pr^(1/3);
    else
        Nu=0.037*Re^.8*Pr^(1/3);
    end 

    %%htc calculation
    htc=(Nu*k)/(fuselageLength);
end 

function internalHTC=getinternalHTC(vent_velocity,density,temperature,pressure)

    %%parameters
    lc=.25;
    mu=double(py.CoolProp.CoolProp.PropsSI("VISCOSITY", "T", temperature, "P", pressure, "Air")); %Pa-s (dynamic viscosity)
    k=double(py.CoolProp.CoolProp.PropsSI("CONDUCTIVITY", "T", temperature, "P", pressure, "Air")); %W/mK (thermal conductivity)
    Cp=double(py.CoolProp.CoolProp.PropsSI("Cpmass", "T", temperature, "P", pressure, "Air")); %J/kgK (specific heat)
    
    %%reynolds and Prandtl number calculations 
    Re=(density*vent_velocity*lc)/mu;
    Pr=(Cp*mu)/k;

    %%Nusselt Correlation calculation
    if Re<5*10^5
        Nu=.664*Re^(1/2)*Pr^(1/3);
    elseif Re>=5*10^5 && Re<3.8*10^6
        Nu=(0.03*Re^.8-871)*Pr^(1/3);
    else
        Nu=0.037*Re^.8*Pr^(1/3);
    end 

    %%htc calculation
    internalHTC=(Nu*k)/(lc);

end 