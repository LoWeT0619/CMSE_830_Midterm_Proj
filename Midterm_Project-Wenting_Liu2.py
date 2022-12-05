###############################
## CMSE 830 Mid-term Project ##
## Wenting Liu               ##
## Oct. 2022                 ##
###############################

# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt

# Title and author name:
st.markdown("## Identification of feature candidates associated with preterm birth through analysis of cord blood data")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Wenting Liu")
with col2:
    st.markdown("### Oct. 2022")
st.text("")

## Workflow:
st.markdown("### 0. The Data Analysis Process")
wf = st.button('Workflow')
if wf:
    st.markdown("""![Workflow](https://raw.githubusercontent.com/LoWeT0619/Fall-2022-CMSE-830/main/Workflow.jpg)""")

# 1. Define the question:
st.markdown("### 1. Define the question")
col1, col2 = st.columns(2)
with col1:
    bg = st.button('Background')
if bg:
    st.markdown("""
    - [**Preterm birth**](https://www.cdc.gov/reproductivehealth/maternalinfanthealth/pretermbirth.htm#:~:text=Preterm%20birth%20is%20when%20a,2019%20to%2010.1%25%20in%202020.) is when a baby is born too early, **before 37 weeks of pregnancy** have been completed. It is known to be associated with chronic disease risk in adulthood. Gestational age (GA) is the most important prognostic factor for preterm infants. In this project, we would like to elucidate which features (include clinical features and cell type features) are related to preterm birth (which is when GA < 37 weeks in this project).

    - With those identified feature candidates, we could then conduct to Epigenome-wide association studies (EWAS). By associating the identified feature candidates with their related DNA information, we could then figure out which gene changed may lead to preterm birth. Then in the future, maybe researchers will find out some targeted medications that can help solve the preterm birth issue or do some risks prediction to make it controllable.
    """)

with col2:
    ques = st.button('The question')
if ques:
    st.latex(r'''
        y = \alpha + \beta_{1} x_{1} + \beta_{2} x_{2} + ... + \beta_{n} x_{n} + \epsilon
    ''')

    st.latex(r'''
        preterm birth = \alpha + \beta_{1} feature_{1} + \beta_{2} feature_{2} + ... + \beta_{n} feature_{n} + \epsilon
    ''')
st.text("")

# 2. Collect the data:
st.markdown("### 2. Collect the data")
col1, col2 = st.columns(2)
with col1:
    data_intro = st.button('Data Introduction')
if data_intro:
    st.markdown("""
    - The dataset comes from the [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7873311/) called *Identification of epigenetic memory candidates associated with gestational age at birth through analysis of methylome and transcriptional data* by Kashima et al. It was published in *Scientific Reports* (The 2021-2022 Journal's Impact is 4.379) in September 2021.
    
    - The paper above shared its related data on [NCBI](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110828).
    
    - We choose one of three datasets from the paper above:
    
        - 110 cord blood samples - Within those 110 cord blood samples, it has: 
        
            - 16 clinical features (6 continuous + 10 discrete): 
                - gestational age
                - birthweight sdscore
                - maternal age
                - maternal bmi
                - paternal age
                - paternal bmi
                - baby gender
                - parity
                - delivery
                - maternal smoking before pregnancy
                - paternal smoking before pregnancy
                - gestational diabetes mellitus
                - chorioamnionitis
                - idiopathic prom without inflammation
                - preeclampsia
                - placenta previa
            
            - 7 cell type features (all continuous):
                - b cell
                - cd4t
                - cd8t
                - gran
                - mono
                - nk
                - nrbc
    """)

## 1.3 Project data:
df = pd.read_csv('https://raw.githubusercontent.com/LoWeT0619/Fall-2022-CMSE-830/main/PB_data.csv')
with col2:
    data = st.button('Data')
if data:
    st.dataframe(df)
st.text("")

# 3. Clean the data:
st.markdown("### 3. Clean the data")

## 3.1 Data Information:
st.markdown("#### 3.1 Data information")
col1, col2 = st.columns(2)
with col1:
    data_desc = st.button('Data Describe')
if data_desc:
    st.write(df.describe())

with col2:
    data_na = st.button('Data Quality - Is there any NA value?')
if data_na:
    st.write(df.isna().sum())

## 3.2 Feature Information:
st.markdown("#### 3.2 Feature Information")
option_l1 = st.selectbox(
    'Select the feature type you want to know:',
    ('clinical features - continuous variable (6/16)',
     'clinical features - discrete variable (10/16)',
     'cell type features - continuous variable (7/7)'))

if option_l1 == 'clinical features - continuous variable (6/16)':
    option_l2_cf = st.selectbox(
        'Select the clinical feature you want to know:',
        ('gestationalAge', 'birthweightSdscore', 'maternalAge', 'maternalBmi', 'paternalAge', 'paternalBmi'))
    col1, col2 = st.columns(2)
    with col1:
        info = df[option_l2_cf].describe()
        st.write(info)
    with col2:
        fig = plt.figure(figsize=(5, 5))
        sns.violinplot(y=df[option_l2_cf])
        st.pyplot(fig)

elif option_l1 == 'clinical features - discrete variable (10/16)':
    option_l2_cf = st.selectbox(
        'Select the clinical feature you want to know:',
        ('babyGender', 'parity', 'delivery', 'maternalSmokingBeforePregnancy',
         'paternalSmokingBeforePregnancy', 'gestationalDiabetesMellitus',
         'chorioamnionitis', 'idiopathicPromWithoutInflammation', 'preeclampsia', 'placentaPrevia'))
    col1, col2 = st.columns(2)
    with col1:
        info = df[option_l2_cf].value_counts()
        st.write(info)
    with col2:
        fig = plt.figure(figsize=(5, 5))
        sns.barplot(x=df[option_l2_cf].value_counts().index, y=df[option_l2_cf].value_counts())
        st.pyplot(fig)
else:
    option_l2_ctf = st.selectbox(
        'Select the cell type feature you want to know:',
        ('estimatedBcell', 'estimatedCd4t', 'estimatedCd8t', 'estimatedGran',
         'estimatedMono', 'estimatedNk', 'estimatedNrbc'))
    col1, col2 = st.columns(2)
    with col1:
        info = df[option_l2_ctf].describe()
        st.write(info)
    with col2:
        fig = plt.figure(figsize=(5, 5))
        sns.violinplot(y=df[option_l2_ctf])
        st.pyplot(fig)
st.text("")

# 4. Analyze the data - IDA & EDA:
st.markdown("### 4. Analyze the data")

# 4.1 IDA - Correlations:
st.markdown("#### 4.1 IDA - Correlations (between continuous variables (6+7)")
col1, col2 = st.columns(2)
with col1:
    corr_hm = st.checkbox('Check correlations among all continuous variables (6+7)')
if corr_hm:
    st.write('Correlation Heatmap')
    sns.set_theme(style="white")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(15, 15))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    st.pyplot(f)

with col2:
    corr_reg = st.checkbox('Check regressions between continuous variables (6+7))')
if corr_reg:
    st.sidebar.title("Regression between two continuous variables")

    labels = df.select_dtypes(include=['float64', 'int64']).columns
    x_axis_choice = st.sidebar.selectbox(
        "x axis",
        labels)
    y_axis_choice = st.sidebar.selectbox(
        "y axis",
        labels)

    min_x = df[x_axis_choice].min(axis=0)
    max_x = df[x_axis_choice].max(axis=0)
    min_y = df[y_axis_choice].min(axis=0)
    max_y = df[y_axis_choice].max(axis=0)

    ############
    ##  Line  ##
    ############
    st.sidebar.write('line: slope and intercept')

    s = float((max_y - min_y) / (max_x - min_x))
    i = float(min_y - s * min_x)

    parameter_list = ['slope', 'intercept']
    parameter_input_values = []
    parameter_default_values = [s, i]
    values = []

    for parameter, parameter_df in zip(parameter_list, parameter_default_values):
        values = st.sidebar.slider(label=parameter, key=parameter, value=float(parameter_df),
                                   min_value=(float(parameter_df) - 3.0), max_value=(float(parameter_df) + 3.0), step=0.1)
        parameter_input_values.append(values)

    input_variables = pd.DataFrame([parameter_input_values], columns=parameter_list, dtype=float)

    slope = float(input_variables['slope'].iloc[0])
    intercept = float(input_variables['intercept'].iloc[0])

    x = np.linspace(min_x, max_x)
    line_df = pd.DataFrame({
        'x': x,
        'y': slope * x + intercept})

    ############
    ## RBF-NN ##
    ############
    st.sidebar.write('RBF: centers, widths and heights')

    c1 = float(min_x)
    w1 = float((min_y - max_y * np.exp(-(min_x - max_x) ** 2)) / (
            1 - np.exp(-(max_x - min_x) ** 2) * np.exp(-(min_x - max_x) ** 2)))

    c2 = float(max_x)
    w2 = float((max_y - min_y * np.exp(-(max_x - min_x) ** 2)) / (
            1 - np.exp(-(min_x - max_x) ** 2) * np.exp(-(max_x - min_x) ** 2)))

    rbf_parameter_list = ['center 1', 'width 1', 'height 1', 'center 2', 'width 2', 'height 2']
    rbf_parameter_input_values = []
    rbf_parameter_default_values = [c1, w1, '0.0', c2, w2, '0.0']
    rbf_values = []

    for rbf_parameter, rbf_parameter_df in zip(rbf_parameter_list, rbf_parameter_default_values):
        rbf_values = st.sidebar.slider(label=rbf_parameter, key=rbf_parameter, value=float(rbf_parameter_df),
                                       min_value=(float(rbf_parameter_df) - 100.0),
                                       max_value=(float(rbf_parameter_df) + 100.0),
                                       step=0.5)
        rbf_parameter_input_values.append(rbf_values)

    rbf_input_variables = pd.DataFrame([rbf_parameter_input_values], columns=rbf_parameter_list, dtype=float)

    center1 = float(rbf_input_variables['center 1'].iloc[0])
    width1 = float(rbf_input_variables['width 1'].iloc[0])
    height1 = float(rbf_input_variables['height 1'].iloc[0])

    center2 = float(rbf_input_variables['center 2'].iloc[0])
    width2 = float(rbf_input_variables['width 2'].iloc[0])
    height2 = float(rbf_input_variables['height 2'].iloc[0])

    x = np.linspace(min_x, max_x)
    rbf_df = pd.DataFrame({
        'x': x,
        'y': width1 * np.exp(-(x - center1) ** 2 / height1 ** 2) + width2 * np.exp(-(x - center2) ** 2 / height1 ** 2)})

    ##############
    ## plotting ##
    ##############
    linear_reg = alt.Chart(line_df).mark_line().encode(
        x='x',
        y='y',
        color=alt.value("#FFAA00"))

    rbf_reg = alt.Chart(rbf_df).mark_line().encode(
        x='x',
        y='y',
        color=alt.value("#00FF00"))

    scatter = alt.Chart(df).mark_circle(size=100).encode(
        x=x_axis_choice, y=y_axis_choice, color='pretermBirth:O',
        tooltip=['gestationalAge', 'birthweightSdscore', 'maternalAge',
                 'maternalBmi', 'paternalAge', 'paternalBmi']).interactive()

    scatter + linear_reg + rbf_reg
st.text("")

# 4.2 EDA - Multivariables Analysis:
st.markdown("#### 4.2 EDA - Multivariables Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    para_cont_cf = st.checkbox('Check relationships among all continuous clinical features (6/16)')
if para_cont_cf:
    # option_l2_cfs = st.multiselect(
    #     'Select the continuous clinical features that you want to explore:',
    #     ['pretermBirth', 'gestationalAge', 'birthweightSdscore', 'maternalAge', 'maternalBmi', 'paternalAge', 'paternalBmi'])
    fig = plt.figure(figsize=(20, 5))
    pd.plotting.parallel_coordinates(df[['pretermBirth', 'gestationalAge', 'birthweightSdscore', 'maternalAge',
                                         'maternalBmi', 'paternalAge', 'paternalBmi']],
                                     'pretermBirth',
                                     color=['blue', 'gray'])
    st.pyplot(fig)

with col2:
    para_dist_cf = st.checkbox('Check relationships among all discrete clinical features (10/16)')
if para_dist_cf:
    # option_l2_cfs = st.multiselect(
    #     'Select the discrete clinical features that you want to explore:',
    #     ['pretermBirth', 'babyGender', 'parity', 'delivery', 'maternalSmokingBeforePregnancy',
    #      'paternalSmokingBeforePregnancy', 'gestationalDiabetesMellitus',
    #      'chorioamnionitis', 'idiopathicPromWithoutInflammation', 'preeclampsia', 'placentaPrevia'])
    fig = plt.figure(figsize=(20, 5))
    pd.plotting.parallel_coordinates(df[['pretermBirth', 'babyGender', 'parity', 'delivery',
                                         'maternalSmokingBeforePregnancy', 'paternalSmokingBeforePregnancy',
                                         'gestationalDiabetesMellitus',
                                         'chorioamnionitis', 'idiopathicPromWithoutInflammation',
                                         'preeclampsia', 'placentaPrevia']],
                                     'pretermBirth',
                                     color=['blue', 'gray'])
    st.pyplot(fig)

with col3:
    para_ctf = st.checkbox('Check relationships among all cell type features (7/7)')
if para_ctf:
    # option_l2_ctfs = st.multiselect(
    #     'Select the cell type features that you want to explore:',
    #     ['pretermBirth', 'estimatedBcell', 'estimatedCd4t', 'estimatedCd8t', 'estimatedGran',
    #      'estimatedMono', 'estimatedNk', 'estimatedNrbc'])
    fig = plt.figure(figsize=(20, 5))
    pd.plotting.parallel_coordinates(df[['pretermBirth', 'estimatedBcell', 'estimatedCd4t', 'estimatedCd8t',
                                         'estimatedGran', 'estimatedMono', 'estimatedNk', 'estimatedNrbc']],
                                     'pretermBirth',
                                     color=['blue', 'gray'])
    st.pyplot(fig)
st.text("")

# 5. Conclusion and discussion:
st.markdown("### 5. Conclusion and discussion")
col1, col2 = st.columns(2)
with col1:
    conclusion = st.button('Conclusion')
if conclusion:
    st.markdown("""
    - We apply the analysis on sigal variable (`violin plot`, `bar plot`), paired variables (`heatmap`, `scatter plot`), and multiple variables (`parallel_coordinates`):
    
    - According the analysis above, our feature candidates associated with preterm birth may be:
    
        - continuous clinical features: **gestational age**
        
        - discrete clinical features: **maternal smoking before pregnancy** 
        
        - cell type features: **cd4t**, **cd8t**, **gran**, **nrbc**
    """)

with col2:
    discussion = st.button('Discussion')
if discussion:
    st.markdown("""
    - Our question has not been solved completely, those figures that we used with can not lead us to the big picture of this project, we analized them seperately. We need conduct to linear or non-linear regression model, like **logistic regression** to explore more.
    
    - As soon as we have the model from above, we could step further to conduct to EWAS analysis, which is the gene level to explore more possibilities.
    """)
st.text("")

