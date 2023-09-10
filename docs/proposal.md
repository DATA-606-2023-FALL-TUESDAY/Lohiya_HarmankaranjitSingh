# Capstone Proposal
 
## 1. Proposal Title: Data Analysis of Crime in Baltimore

- **Author Name** - Harmankaranjit Singh Lohiya 

- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- [GitHub](https://github.com/KARANS12)
- [LinkedIn](www.linkedin.com/in/harmankaranjit-singh-b33161251)
- **Link to your PowerPoint presentation file** - In Progress
- **Link to your YouTube video** - In Progress


## 2. Background


"Crime in Baltimore" dataset is related to crime incidents in the city of Baltimore. It contains information about various crime-related attributes for different incidents and it's main agenda will be how it's gonna be a direct impact on public safety, resource allocation, policy decisions, community engagement, and the overall well-being of a city's residents. By analyzing and acting upon this data, cities can work to reduce crime rates, improve safety, and enhance the quality of life for their communities.

### Now, why does it matter?

Public Safety: Crime data analysis is essential for ensuring the safety of residents and visitors in a city. By identifying crime patterns and trends, law enforcement agencies can allocate resources effectively, deploy officers to high-crime areas, and take proactive measures to prevent criminal activities.

Resource Allocation: It helps governments and law enforcement agencies allocate resources efficiently. By analyzing crime hotspots and types, they can make informed decisions about where to allocate police officers, surveillance, and other resources to deter criminal activities.

Crime Prevention: Insights from crime data can lead to more effective crime prevention strategies. Identifying recurring patterns can help design interventions and programs aimed at reducing specific types of crimes or criminal behavior.

Community Engagement: Crime data can be shared with the community, fostering transparency and citizen engagement. Communities can become more actively involved in neighborhood watch programs, community policing efforts, and other initiatives aimed at reducing crime.

Policy Development: Policymakers can use crime data to inform the development of public policies related to crime prevention, criminal justice reform, and public safety. Data-driven policies are often more effective and targeted.

Resource Efficiency: Efficient allocation of resources based on data analysis can lead to cost savings for municipalities. By targeting resources where they are needed most, cities can reduce unnecessary expenditures.

Public Perception: Crime rates and trends can influence the perception of safety within a city. A reduction in crime can improve the quality of life for residents and encourage economic development.

Criminal Justice: The criminal justice system benefits from crime data analysis by identifying trends in types of cases, case clearance rates, and areas where additional court resources may be needed.

Research and Academia: Crime data provides a valuable resource for researchers and academics studying criminology, sociology, and related fields. It enables the development of theories, evaluation of interventions, and the advancement of knowledge in these areas.

Emergency Response: Crime data can also inform emergency response planning. Understanding when and where crimes are more likely to occur can help emergency services prepare for potential incidents.

### Research Scope?

Crime Pattern Analysis:
Analyzing historical crime data to identify recurring patterns, trends, and spatial-temporal variations in crime occurrences.
Investigating the factors influencing crime patterns, such as socio-economic conditions, population density, and environmental factors.

Predictive Modeling:
Developing predictive models to forecast future crime rates, types, and locations based on historical data and relevant features.
Evaluating the effectiveness of predictive policing strategies in preventing and responding to crimes.

Crime Severity Assessment:
Creating models or algorithms to assess the severity of different types of crimes and prioritize law enforcement responses accordingly.
Investigating the impact of crime severity on the criminal justice system and sentencing outcomes.

Geospatial Analysis:
Conducting geospatial analysis to map and visualize crime hotspots, identifying areas with elevated crime rates.
Assessing the effectiveness of spatially targeted interventions and community policing efforts.

Community Engagement and Policing Strategies:
Studying community engagement initiatives, neighborhood watch programs, and their impact on crime prevention.
Evaluating the effectiveness of community-oriented policing strategies in building trust and reducing crime.

Social and Demographic Factors:
Analyzing the relationship between crime rates and socio-demographic factors, including income, education, race, and employment.
Investigating the role of social determinants of crime and inequality.

Crime Prevention and Intervention Programs:
Evaluating the outcomes of crime prevention programs and interventions, such as youth mentorship, drug rehabilitation, and restorative justice programs.
Assessing the cost-effectiveness and long-term impact of these initiatives.

Criminal Justice System Analysis:
Investigating the processing of criminal cases within the criminal justice system, including case clearance rates, court outcomes, and sentencing disparities.
Identifying areas for reform and improvements in the justice system.

Temporal Analysis:
Studying temporal patterns in crime, including daily, weekly, and seasonal variations.
Examining the influence of time-related factors on crime commission and reporting.

Machine Learning and AI Applications:
Exploring the application of machine learning and artificial intelligence techniques for crime prediction, anomaly detection, and risk assessment.
Developing predictive policing tools and real-time crime monitoring systems.

Public Policy and Governance:
Assessing the impact of public policies, such as gun control measures, policing practices, and criminal justice reforms, on crime rates and public safety.
Providing evidence-based recommendations for policy development.

Data Privacy and Ethics:
Addressing ethical considerations in the collection, use, and sharing of crime data, especially in the context of privacy concerns and data security.

Interdisciplinary Research:
Collaborating across disciplines, including criminology, sociology, data science, urban planning, and public health, to gain a comprehensive understanding of crime dynamics.

### Few Research questions could be: 

1. What are the ethical implications of collecting and using crime data, and how can data privacy concerns be addressed while ensuring public safety?
2. How can transparent and accountable data practices be implemented in the context of crime data analysis?
3. Can machine learning models accurately predict the likelihood of specific types of crimes occurring in different neighborhoods and at different times?
4. How can predictive policing models be effectively integrated into law enforcement strategies to prevent crimes and allocate resources efficiently?
5. How can crime severity be quantified and categorized, and what are the implications of using different severity metrics on law enforcement priorities?
6. Does the severity of a crime impact the probability of arrest, conviction, or sentencing outcomes?
7. What are the correlations between crime rates and socio-demographic factors, and how do these relationships vary by crime type?
8. To what extent does income inequality contribute to crime disparities within Baltimore neighborhoods?

## 3. Data 

**Describe the datasets you are using to answer your research questions.**

- Data sources: https://www.kaggle.com/datasets/sohier/crime-in-baltimore
- Data size: 9 MB
- Data shape: # of rows - 89943 and # columns - 15
 


### **What features are important, what column means what**

CrimeDate: The date when the crime occurred.

CrimeTime: The time of day when the crime occurred.

CrimeCode: A code for the type of crime.

Location: The location where the crime took place.

Description: A description of the crime.

Inside/Outside: Indicates whether the crime occurred inside or outside.

Weapon: The type of weapon used.

Post: Police post associated with the incident.

District: The police district in Baltimore.

Neighborhood: The neighborhood where the crime occurred.

Longitude and Latitude: Geographic coordinates of the crime location.

Location 1: Additional geographic location information.

Premise: The type of premise where the crime occurred.

Total Incidents: Possibly the total number of incidents at that location or within a specific timeframe.
