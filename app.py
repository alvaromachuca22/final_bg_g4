#import nltk
import pickle 
import streamlit as st
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS



modelo_RL_TFIDF = pickle.load(open('LogisticRegression-TF-IDF.pkl', 'rb'))





vectorizer = pickle.load(open('vectorizer_BD.pkl', 'rb'))






def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(token)
    return result


# --- MODELO LDA --- 
modelo_lda_BD = pickle.load(open('modelo_lda_BD.pkl', 'rb'))
dictionary_BD = pickle.load(open('dictionary_LDA_BD.pkl', 'rb'))



def predict_tweet(tweet):

    labels = ['negative','positive']
    
    nuevo_tweet = np.array([tweet])
    
    nuevo_tweet_vector1 = vectorizer.transform(nuevo_tweet)

    prediction1 = modelo_RL_TFIDF.predict_proba(nuevo_tweet_vector1)[0]

    pred1_1 = list(modelo_RL_TFIDF.predict(nuevo_tweet_vector1)) 

    pred1_1 = labels[np.argmax(pred1_1)]

    pred1_2 = str(round(max(prediction1)*100,2)) +'%'


    #-------------------------LDA-----------------------------------------------------

    diccionario = {0: 'Personas desean tratamiento para poner fin a la pandemia', 
               1: 'Atención a pacientes con coronavirus y conocimiento sobre prevención para no contraer coronavirus', 
               2: 'Compras de vacunas por parte del gobierno para el retorno a las escuelas para niños y niñas',  
               3: 'Medidas de prevención para continuar con el trabajo en tiempos de pandemia', 
               4: 'Medidas planteadas por los gobiernos para enfrentar al coronavirus', 
               5: 'Noticias sobre incremento de casos de muertes por coronavirus ', 
               6: 'Noticias sobre la lucha para preservar la vida de las personas con coronavirus', 
               7: 'Busqueda de una vacuna efectiva contra el nuevo coronavirus y confinamiento de niños en tiempos de pandemia', 
               8: 'Noticias sobre los hospitales del Ministerio de salud para el cuidado de pacientes con Covid-19', 
               9: 'Conocimiento de las personas acerca de la cuarentena general ', 
               10:'Desarrollo de nuevos estudios y creación de vacuna contra el Covid-19', 
               11:'Informe sobre nuevos casos positivos de coronavirus'
               }



    bow_vector = dictionary_BD.doc2bow(tweet.split())

    for index, score in sorted(modelo_lda_BD[bow_vector], key=lambda tup: -1*tup[1]):

        for key,value in diccionario.items():
            if index == key:
                name = value
                break
    
    #print("\nTopic: {} \nScore: {}\t \nWords: {}".format(name,score, modelo_lda_BD.print_topic(index, 5)))
    
        pred_lda = "El tweet pertenece al tópico" + " " + "'" + name + "'"  + " " + "con una probabilidad de" + " " + str(round(score*100,2)) + "%" 
    
        break

    #-------------------------HDP-----------------------------------------------------

    




    return pred1_1, pred1_2, pred_lda



def main():
    
    ###st.title("Trabajo final Big Data - Grupo 4")

  
    html_temp = """
    <div>
    <p> Proyecto final de Big Data - Grupo 4 </p>
    </div>

    <div style="background-color:#025246 ;padding:10px">

    <h2 style="color:white;text-align:center;">Sistema de Análisis de sentimiento y detección de tópicos usando keywords relacionados al Covid-19</h2>
    
    </div>

    
    <!--
    <form action="{{ url_for('predict')}}"method="post">
        <input type="text" name="rate" placeholder="rate" required="required" />
        <input type="text" name="sales in first month" placeholder="sales in first month" required="required" />
        <input type="text" name="sales in second month" placeholder="sales in second month" required="required" />
        <button type="submit" class="btn btn-primary btn-block btn-large">Predict sales in third month</button>
    </form>
    
    -->

    """  
    
    st.markdown(html_temp, unsafe_allow_html = True)
    

    tweet = st.text_input("", "Escribe tu tweet aquí")
    

    
    if st.button("Predecir"):
        output = predict_tweet(tweet)
        #st.success('El tweet ingresado es{}'.format(" " + str(output[0])) + "  " + 'con una probabilidad de{}'.format(" " + str(output[1])))
        st.markdown('Resultado de **Análisis de sentimiento** :')


        st.markdown('Según mejor modelo de Machine learning: Regresión Logística - TF-IDF')
        st.success('El tweet ingresado es{}'.format(" " + str(output[0]).capitalize() ) + "  " + 'con una probabilidad de{}'.format(" " + str(output[1])))


        st.markdown('Resultado de **Topic Modeling** :')

        st.markdown(' Según modelo de detección de tópicos: LDA')
        st.success(str(output[2]))

        #st.success(str(output[3]))



    html_temp2 = """
    
    <h1 style = "color:black; text-align:center;"> Evolución del total de casos de fallecidos por Covid-19</h1>
    <iframe src="https://ourworldindata.org/grapher/total-deaths-covid-19?country=ITA+ESP+USA" style="width: 100%; height: 600px; border: 0px none;"></iframe>

    <h1 style = "color:black; text-align:center;"> Total de pruebas rápidas para detección de Covid-19 por cada 1000 personas</h1>
    <iframe src="https://ourworldindata.org/grapher/full-list-cumulative-total-tests-per-thousand" style="width: 100%; height: 600px; border: 0px none;"></iframe>



    <h1 style = "color:black; text-align:center;"> Crecimiento de casos confirmados de casos de Covid-19</h1>
    <iframe src="https://ourworldindata.org/grapher/covid-confirmed-cases-since-100th-case?country=USA+ESP+ITA+KOR+TWN+GBR+NOR" style="width: 100%; height: 600px; border: 0px none;"></iframe>
    

    <h1 style = "color:black; text-align:center;"> Mapa de casos confirmados de Covid-19 en el mundo</h1>
    <iframe src="https://ourworldindata.org/grapher/total-cases-covid-19?tab=map" width="100%" height="600px"></iframe>


    <h1 style = "color:black; text-align:center;"> Total de casos confirmados por coronavirus por contimente</h1>
    <iframe src="https://ourworldindata.org/grapher/total-cases-covid-19?tab=map&region=Africa" width="100%" height="600px"></iframe>

    <br><br/>

    """

    st.markdown(html_temp2, unsafe_allow_html = True)



    st.markdown('**¿Deseas escuchar música?**')

    #st.markdown('Instrumental de piano')
    #audio_file = open('instrumental_piano.mp3', 'rb')
    #audio_bytes = audio_file.read()
    #st.audio(audio_bytes, format='audio/mp3', start_time = 0)


    st.markdown('Sonata Claro de luna - Beethoven')
    audio_file2 = open('beethoven.mp3', 'rb')
    audio_bytes2 = audio_file2.read()
    st.audio(audio_bytes2, format='audio/mp3', start_time = 0)



if __name__ == '__main__':
    main()