%%This is the test script for thermal analysis 
clear; clc;
latitude=37; %degrees
solarTime=12; %hours
altitude=21; %km
airSpeed=20; %m/stime
totalTime=86400; %%seconds
vent_velocity=0;
[internalHTC,airTemperature,temperatureTrackArray, qnetArray, outCondArray, convectionArray, ...
          absSolarIrradianceArray, radiosityArray, insideConductionArray, ...
          timeArray, payloadTemp, payloadNetPower, powerbrickTemp, powerbrickNetPower, ...
          auterionTemp, auterionNetPower, gpsTemp, gpsNetPower, ethernetTemp, ethernetNetPower, ...
          radioTemp, radioNetPower, battery1Temp, battery1NetPower, battery2Temp, battery2NetPower, ...
          battery3Temp, battery3NetPower, battery4Temp, battery4NetPower, ...
          dTpayload, dTpowerbrick, dTauterion, dTgps, dTethernet, ...
          dTradio, dTbattery1, dTbattery2, dTbattery3, dTbattery4, ...
          dTwallinside, dTwalloutside,TwallinsideTemp, TwallinsideNetPower, TwalloutsideTemp, Qwallout,silvusTemp,silvusNetPower] ...
          = thermalAnalysis(altitude, latitude, 152, solarTime, airSpeed, totalTime,vent_velocity);


%% plotting 


%%solar load 
figure(1)
plot(timeArray,absSolarIrradianceArray)
hold on 
plot(timeArray,outCondArray)
plot(timeArray,-convectionArray)
plot(timeArray,-radiosityArray)
plot(timeArray,Qwallout)
title("heat transfer methods @outside of wall")
xlabel("Time (hrs)")
ylabel("Heat (W)")
legend("Solar Irradiance", "Conduction", "Convection","Radiosity", "Total Heat Flux")
hold off 

%% temperatures of avionics 
figure(2)
plot(timeArray,payloadTemp)
hold on 
plot(timeArray,powerbrickTemp)
plot(timeArray,auterionTemp)
plot(timeArray,gpsTemp)
plot(timeArray,ethernetTemp)
plot(timeArray,radioTemp)
plot(timeArray,silvusTemp)
title("Temperature of each componenet (K)")
xlabel("Time (hrs)")
ylabel("Temperature (K)")
legend("Payload","Power Brick","Auterion","GPS","Ethernet","Radio", "Silvus")
hold off 

%% net power due to radiosity netwrok
figure(3)
plot(timeArray,battery1NetPower)
hold on 
plot(timeArray,battery2NetPower)
plot(timeArray,battery3NetPower)
plot(timeArray,battery4NetPower)
plot(timeArray,TwallinsideNetPower)
title("Heat in/out due to radiation (W)")
xlabel("Time (hrs)")
ylabel("Power (W/m)")
legend("Battery1","Battery2","Battery3","Battery4","Wall Inside")
hold off 

%% plot dT/dt for each component
figure(4)
plot(timeArray,dTpayload);
hold on
plot(timeArray,dTpowerbrick)
plot(timeArray,dTauterion)
plot(timeArray,dTgps)
plot(timeArray,dTethernet)
plot(timeArray,dTradio)
title("Rate of Temperature Change (dT/dt) for Each Component")
xlabel("Time (hrs)")
ylabel("dT/dt [K/s]")
legend("Payload","Power Brick","Auterion","GPS","Ethernet","Radio")
hold off

%% temperature of batteries 
figure(5)
plot(timeArray,battery1Temp)
hold on 
plot(timeArray,battery2Temp)
plot(timeArray,battery3Temp)
plot(timeArray,battery4Temp)
plot(timeArray,TwalloutsideTemp)
plot(timeArray,TwallinsideTemp)
title("Temperature of each battery (K)")
xlabel("Time (hrs)")
ylabel("Temperature (K)")
legend("Battery1","Battery2","Battery3","Battery4","Wall Outside","Wall Inside")
hold off 



%% dT/dt of batteries 
figure(6)
hold on 
plot(timeArray,dTbattery1)
plot(timeArray,dTbattery2)
plot(timeArray,dTbattery3)
plot(timeArray,dTbattery4)
plot(timeArray,dTwallinside)
plot(timeArray,dTwalloutside)
title("dT/dt of each battery (K)")
xlabel("Time (hrs)")
ylabel("dT/dt [K/s]")
legend( "Battery1","Battery2","Battery3","Battery4","Wall Inside","Wall Outside")
hold off

%% net power due to radiosity netwrok
figure(7)
hold on 
plot(timeArray,payloadNetPower)
plot(timeArray,powerbrickNetPower)
plot(timeArray,auterionNetPower)
plot(timeArray,gpsNetPower)
plot(timeArray,ethernetNetPower)
plot(timeArray,radioNetPower)
plot(timeArray,silvusNetPower)
title("Heat in/out due to radiation (W)")
xlabel("Time (hrs)")
ylabel("Power (W/m)")
legend("Payload","Power Brick","Auterion","GPS","Ethernet","Radio","Silvus")
hold off