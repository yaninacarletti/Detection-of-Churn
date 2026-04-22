Proyecto de Predicción de Churn



🚀 Descripción General

Este proyecto tiene como objetivo desarrollar un modelo de machine learning para predecir el churn de clientes a partir de datos de comportamiento.

La idea es identificar usuarios con alta probabilidad de abandono para poder implementar estrategias de retención de forma proactiva.




🧠 Definición del Problema

El churn se define como:

Un usuario que no presenta actividad en los siguientes 5 días a partir de una fecha de observación (snapshot_date).

Este enfoque permite:

- Evitar data leakage
- Simular un escenario real de negocio
- Generar predicciones accionables




🗂️ Descripción del Dataset

El dataset contiene información de uso por usuario:

- user_id
- date
- actions_performed
- time_spent_minutes
- documents_created
- logins
- signup_date
- plan_type
- country



  
⚙️ Metodología
1. 📅 Enfoque basado en Snapshot

Se definió una snapshot_date y se dividieron los datos en:

- Pasado (df_past) → para construir features
- Futuro (df_future, 5 días) → para definir el churn

  
2. 🎯 Construcción del Target
churned = 1 → usuario inactivo en la ventana futura
churned = 0 → usuario activo


3. 🧩 Feature Engineering

Se generaron variables agregadas por usuario:

🔹 Métricas de actividad
- total de acciones
- tiempo promedio de uso
- documentos creados
- número de logins
  
🔹 Features de comportamiento
- actions_per_day
- time_per_login
  
🔹 Features temporales
- days_since_last_activity (recency)
- days_since_signup (tenure)
  
🔹 Variables categóricas
- One-hot encoding de plan_type y country

  
4. 🧪 División Train/Test

Se utilizó un train-test split estratificado, adecuado dado que el dataset ya respeta la temporalidad gracias al enfoque de snapshot.


5. ⚖️ Manejo del Desbalance de Clases

En lugar de aplicar oversampling (SMOTE), se utilizó:

- class_weight='balanced' en:
  - Logistic Regression
  - Random Forest
- scale_pos_weight en:
  - XGBoost



    
🤖 Modelos Utilizados
- Logistic Regression
- Random Forest
- XGBoost



  
📊 Métricas de Evaluación

Dado el desbalance de clases, se priorizaron:

- Recall
- Precision-Recall AUC
- ROC-AUC



   
📈 Resultados
Modelo	                           ROC-AUC       /     	PR-AUC
- Logistic Regression                	~0.43             	~0.07
- Random Forest	                      ~0.56             	~0.40
- XGBoost	                            ~0.66              	~0.41




🛠️ Tecnologías Utilizadas
- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn




🔍 Insight principal

XGBoost obtuvo el mejor desempeño
Aún hay margen de mejora en la capacidad predictiva.


La definición del target es crítica en problemas de churn
Evitar data leakage es fundamental
El recall es más importante que la accuracy en este tipo de problemas
El contexto de negocio debe guiar la evaluación del modelo


