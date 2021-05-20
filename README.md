# VehicleReversing_IITGn_IIQ


* ## Instructions to run the reversing detector - >

      python3 reversing_detector.py --video=truck_reversing.mp4 

* When reversing for some object is detected, color of label of that object changes from 
  ### GREEN to RED.
* ## Problem Statement:
To detect vehicle reversing at a T-junction and raise an alert signal if any human was present in that area. The problem statement suggests that insufficient view and hence lack of judgement makes it difficult for truck drivers to reverse the trucks in large manufacturing industries. On top of this, if a driver was even a bit careless or not attentive while reversing the truck, and someone behind the truck was also not attentive, perhaps because he was busy with his/her work or couldn’t realize the presence of a truck reversing, then this could lead to a severe mishap. If the frequency of trucks moving in and out of the industry is large, then the probability of occurrence of such an event increases, and which can be dangerous because its consequence could be as bad as loss of life. Hence we are required to design an alert system that will raise an alarm whenever reversing happens at the T-joint so that everyone is aware of the same, and hence safe.

* ## Pseudo Code for Centroid tracker:
* ## Pseudo Code for Reversing detection
* ## Results:
     As the truck starts to reverse, label changes from green to red </br>
     [Output video](https://drive.google.com/file/d/1LQa1HwsG1Zy99FlIjpyexHj5GHMzz9dD/view?usp=sharing)
* ## Limitations:
  1. Between the previous halt and the next halt, if there are not enough sets of n frames, then we might not be able to calculate the correct direction of the object before the next halt.
  2. Not multi-threaded, but can be done.

* ## Future work:
  Setting up an Alarm system, i.e. deploying the algorithm on some edge device.
