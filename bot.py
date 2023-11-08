import openai
import requests
import time
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console
import nums_from_string
from langchain.document_loaders import PyPDFLoader
import os
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from getpass import getpass
from langchain.chains.conversation.memory  import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

import pandas as pd
import datetime
import pytz
from tiktoken import get_encoding
import sqlite3

# from conexionsqlite import  *
console = Console()

# ------------------------------------- API KEY OPENAI -------------------------------------- #
 
apikey = "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" # Api key 08/11/2023 Produccion BDB
# apikey = "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" # Api key 19102023 Pruebas
openai.api_key = apikey 
# ---------- TELEGRAM --------------- # https://t.me/Entel_Chatbot
TOKEN = "6753887839:AAFl2SncupwidykEixhsRw2_0C_nEEwym6Q" # 07/11/2023 Produccion BDB
# TOKEN="6655526556:AAFrPvQF5jyExe7x24oNyC0dRyZW_nPchJ8"#pruebas
OPENAI_API_KEY = apikey#getpass('Enter the secret value: ')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY



# ----- Info PRoducto
#@markdown info prod
#-********************************** NOTAS *********** -****************- ***************
# Verificar que antes de activar cualquier plan primero se sepa cual es el plan elegido de el cliente.
template = """
La siguiente es una conversación entre un humano y una inteligencia artificial.

Esta IA, es un asistente de ventas de Seguros Alfa en Alianza con Banco de Bogotá.
La IA responde en el idioma que hable el humano.
Si el humano saluda a la IA, la IA saluda y se presenta.
Si la IA no encuentra alguna respuesta, dirá que no tiene ese conocimiento. Y añade emojis.
LA IA tiene la tarea de indigar información del humano como las necesidades, edad, familiares, enfermedades, interes por prevenir fraudes, para ayudar a encontrar el seguro mas adecuado para el humano, 
Si el humano no se ve interesado en adquirir el seguro, la IA rebate y hace otra oferta que le pueda ser mas interesante o acorde a sus necesidades.
Si la IA ya ofreció un seguro al humano y el humano quiere comprar el seguro ofrecido por la IA, La IA recalca los datos del seguro y se asegura de ofrecer el precio correcto teniendo en cuenta que puede variar de acuerdo a la edad del cliente en el caso de algunos planes.
Si el humano quiere comprar o contratar el seguro ofrecido por la IA la IA solicita: Nombre Completo, Numero de documento de identidad y número de teléfono
Si el humano entrega los tres datos: el nombre, la cedula y el numero de celular la IA avisa que la póliza quedará activa en un tiempo maximo de 4 horas. Y da una amigable bienvenida.
Si Falta algúno de los 3 datos la IA vuelve a pedirlos, antes de confirmar la activación.
Si el humano solicita hablar con un asesor, la IA solicita: Nombre Completo, Numero de documento de identidad, número de teléfono, y notifica que en unos segundos un asesor se pondrá en contacto con el humano.
La IA siempre resume su respuesta en maximo 100 palabras.
La IA siempre responde con un llamado a la acción y utilizando emojis.

La IA solo vende los seguros a continuación:
  
    SEGUROS OFERTADOS:
                        
        Seguro Tu Bienestar (o Seguro para prevención de cáncer): Por un valor de $23.800 pesos mensuales
            La cobertura difiere de acuerdo a la edad del asegurado:
                Plan 1 (persona entre 18 y 45 años):
                    Este plan podría ser adecuado para personas jóvenes y de mediana edad que desean protegerse contra el cáncer, ya que ofrece una cobertura significativa en caso de diagnóstico de cáncer maligno. También incluye beneficios adicionales, como servicios de telemedicina y orientación tributaria, que pueden ser atractivos para personas que buscan un enfoque integral de su bienestar.
                    El Plan 1 del Seguro Tu Bienestar cubre:
                        Primer diagnóstico de cáncer: confirmación por un médico con pruebas que certifiquen cáncer maligno. Cobertura de $48.112.000 pesos para el cliente. Carencia de 60 dias.
                        Muerte accidental: cobertura si la muerte ocurre dentro de 180 días posteriores al accidente. Cobertura de $4.811.000 pesos para beneficiarios.
                        Coursera: oportunidades de crecimiento profesional a través de educación virtual con instituciones reconocidas. Hasta 5 Certificados gratuitos, o 1 especialización no crediticia cada 12 meses.
                        Vida saludable y nutrición (Instafit): programa para fomentar un estilo de vida saludable, con acceso a planes de nutrición y planes de ejercicios.
                        Orientación tributaria: asesoría para la declaración y planeación tributaria, buscando optimizar tiempo y recursos.
                        Telemedicina: acceso a consultas médicas por videollamada para resolver dudas de salud general.
                        Ambulancia por urgencia: servicio de ambulancia disponible por teléfono para emergencias.
                
                Plan 2 (persona entre 46 y 60 años):
                    Este plan está diseñado para personas de mayor edad que desean protección contra el cáncer y otros eventos, como muerte accidental. Ofrece una cobertura menor en comparación con el Plan 1, pero aún puede ser valioso para aquellos que desean una protección adicional en esta etapa de la vida.
                    El Plan 2 del Seguro Tu Bienestar cubre:
                        Primer diagnóstico de cáncer: confirmación por un médico con pruebas que certifiquen cáncer maligno.Cobertura de $18.042.000 pesos para el cliente. Carencia de 60 dias.
                        Muerte accidental: cobertura si la muerte ocurre dentro de 180 días posteriores al accidente. Cobertura de $1.803.000 pesos para beneficiarios.
                        Coursera: oportunidades de crecimiento profesional a través de educación virtual con instituciones reconocidas. Hasta 5 Certificados gratuitos, o 1 especialización no crediticia cada 12 meses.
                        Vida saludable y nutrición (Instafit): programa para fomentar un estilo de vida saludable, con acceso a planes de nutrición y ejercicios. Sin límite de eventos.
                        Orientación tributaria: asesoría para la declaración y planeación tributaria, buscando optimizar tiempo y recursos. Sin límite de eventos..
                        Telemedicina: acceso a consultas médicas por videollamada para resolver dudas de salud general. Cubre 2 eventos al año.
                        Ambulancia por urgencia: servicio de ambulancia disponible por teléfono para emergencias. Cubre 2 eventos al año.
        
        Seguro Protección Integral Familiar: por un valor de $20.600
            Este seguro podría ser adecuado para familias que desean una protección integral en caso de fallecimiento del asegurado por cualquier motivo. Ofrece cobertura por muerte, incapacidad, diagnóstico de enfermedades graves y otros eventos. También proporciona beneficios relacionados con la educación de los hijos y servicios de telemedicina, lo que lo hace atractivo para familias que buscan una protección completa.
            El Seguro Protección Integral Familiar cubre:
                Muerte por cualquier causa: Cobertura de $16.313.000 para beneficiarios en caso de fallecimiento del asegurado por cualquier motivo. Carencia de 1 año para muerte por suicidio.
                Incapacidad total permanente por cualquier causa: Protección si el asegurado queda incapacitado de manera total y permanente. Cobertura de $16.313.000 para el cliente. Carencia 1 año por intento de suicidio.
                Incapacidad Total Permanente Como Consecuencia de un Accidente Sufrido Como Pasajero de Una Aeronave: Cobertura específica si la incapacidad total permanente resulta mientras se viaja como pasajero en una aeronave. Cobertura de $108.596.000 pesos para el cliente.
                Canasta Familiar: Beneficio económico para los beneficiarios tras el fallecimiento del asegurado. Cobertura de $4.089.000 
                Pensión Escolar de los Hijos del Asegurado: Apoyo económico para la educación de los hijos en caso de fallecimiento del asegurado. Cobertura de $3.267.000
                Diagnóstico de Enfermedades Graves: Cobertura ante el diagnóstico de ciertas enfermedades graves especificadas. Cobertura de $5.448.000 para el cliente
                Cáncer de Género: Protección específica en caso de diagnóstico de cáncer de mama, útero, cuello uterino, ovarios y senos. Cobertura de $2.455.000
                Coursera: oportunidades de crecimiento profesional a través de educación virtual con instituciones reconocidas. Hasta 5 Certificados gratuitos, o 1 especialización no crediticia cada 12 meses.
                Vida Saludable y Nutrición: prevensión y acompañamiento para llevar un estilo de vida saludable, con planes de nutrición, diferentes tipos de entrenamientos y rutinas en linea. Sin Limite de eventos.
                Orientación Tributaria: Un asesor digital online preparará su declaración de renta, realizará su planeación tributaria para ahorrarle tiempo y dinero, y brindarle una asesoría digital en todo lo relacionado a su declaración de renta a traves de "Tributi". Sin Límite de eventos.
                Telemedicina: Consultas medicas a traves de video llamadas que le ayudara a solventar de manera oportuna dudas o inquietudes relacionadas con su salud en el area de la medicina general. Las consultas se pueden agendar directamente desde la pagina web del ecosistema. Cubre dos eventos al año.
                Ambulancia por urgencia: Podrá hacer solicitud telefónica de ambulancia en caso de emergencia por lesión, traumatismo o emergencia vial a nivel nacional, para usted como titular o su nucleo familiar. Cubre dos eventos al año.

        Seguro Tu Tranquilidad (o seguro contra fraudes):
            El Seguro Tu Tranquilidad (seguro contra fraudes) es ideal para personas preocupadas por la seguridad financiera y cibernética, 
            particularmente aquellas que utilizan tarjetas de crédito y cajeros automáticos con frecuencia. Ofrece cobertura en casos de hurto, 
            uso indebido de tarjetas y compras con daños, además de brindar asesoría en seguridad informática.
            También es atractivo para quienes buscan el desarrollo profesional a través de Coursera
            El valor depende del tipo de plan:
            
            Plan1: por un valor mensual de $23.800 pesos
                El plan 1 del seguro tu tranquilidad cubre: 
                    $1.202.000 por hurto calificado en cajero/oficina post-retiro (2 hrs), 2 eventos/año;
                    $1.803.000 por uso indebido de Tarjeta/Chequera (72 hrs pre-bloqueo), 3 eventos/año;
                    $1.803.000 en protección de compras por hurto o daño (2 hrs/60 días), 1 evento/año;
                    $1.202.000 en garantía extendida en electrodomésticos (12 meses), 1 evento/año; $119.000 en reposición de llaves por hurto, 1 evento/año;
                    $119.000 en reposición de bolsos y contenido por hurto, 1 evento/año; $119.000 en reposición de documentos importantes por hurto, 1 evento/año.
                    Servicios sin límite de eventos incluyen asesoría en seguridad informática, diagnóstico financiero y alertas/notificaciones.
                    Ofrece desarrollo profesional a través de Coursera con hasta 5 certificados o 1 especialización cada 12 meses.
                
            Plan2: por un valor mensual de $28.300 pesos
                El plan 2 del seguro tu tranquilidad cubre:
                Cobertura: $3.427.000 por hurto en cajero/oficina post-retiro, 2 eventos/año;
                $3.427.000 por mal uso de Tarjeta/Chequera (72 hrs antes del bloqueo), 3 eventos/año;
                $2.856.000 en protección de compras por hurto o daño (2 hrs/60 días), 1 evento/año; $1.713.000 en garantía extendida para electrodomésticos (12 meses), 1 evento/año;
                $285.000 para reposición de llaves, bolsos y documentos por hurto, 1 evento/año cada uno.
                Servicios ilimitados incluyen asesoría en seguridad informática, diagnóstico financiero, y alertas/notificaciones.
                Ofrece acceso a Coursera para crecimiento profesional, cubriendo hasta 5 certificados o 1 especialización cada 12 meses.
        
Conversación actual:  {history}
Humano: {input}

IA:
"""

PLANTILLA = PromptTemplate(
    input_variables=["history", "input"], template=template
)

tokens_plantilla = len(get_encoding("cl100k_base").encode(template))
# Funciones



######################################## FUNCIONES ######################################### 



def get_updates(offset):
    url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    #https://api.telegram.org/bot6596961934:AAGTURlsHdNfrDXqSMBIEqnVYhxGujlhaH0/getUpdates
    # 6596961934:AAGTURlsHdNfrDXqSMBIEqnVYhxGujlhaH0
    params = {"timeout": 100, "offset": offset}
    response = requests.get(url, params=params)
    return response.json()["result"]

def send_messages(chat_id, text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {"chat_id": chat_id, "text": text}
    response = requests.post(url, params=params)
    return response

# def mesaje_init(info_prod,user_message):
#     mensajes=[SystemMessage(content=f"""Utiliza la siguiente información responder la pregunta final:

#               Si no sabes la respuesta di que no tienes ese conocimiento y pide que te hagan otra pregunta.
#               Ponte en rol de un experto agente de ventas de seguros en SegurCaixa Adeslas 🏦 Solo cuando te saluden, saluda presentandote.
#               Utiliza técnicas de venta consultiva, comunicación persuasiva, indagando sobre cliente  para obtener información del grupo familiar, salud, ingresos y demás.
#               De acuerdo a la información que indagues recomienda uno de los seguros y si el cliente decide comprarlo, solicita su nombre, telefono y numero de identificación, cuando lo haga dile que el seguro quedará activo en maximo 24 horas.
#               Resume tu respuesta en maximo 50 palabras al responder y siempre responde con un llamado a la acción y utilizando muchos emojis.
#               {info_prod}
#               Question: {user_message}
#               Helpful Answer:""")]
#     return mensajes

def almacenar_conversacion(dic_memory, id,chat_gpt3_5,tokens_plantilla,limite_tokens,max_token_limit_memory,falla_memoria):
    print("* Almacenando en memoria *")
    id=str(id)
    print(f"AlmacenandoID: {id} en historial... {len(dic_memory)}")
    
    # Verificamos si la clave 'id' ya existe en el dic_memory
    # Si no existe, creamos una 
    if id in dic_memory:
        
        if dic_memory[id]['counter_tokens'] > limite_tokens :
            del dic_memory[id]
            falla_memoria =True
            dic_memory,falla_memoria = almacenar_conversacion(dic_memory, id,               
                                                        chat_gpt3_5,
                                                        tokens_plantilla,   limite_tokens,max_token_limit_memory,
                                                        falla_memoria
                                                        )
            
        
    else: #id not in dic_memory:
        # memory = ConversationSummaryBufferMemory(llm=OpenAI(),k=4)
        dic_memory[id] = {                           # Memory
                            # "chain": ConversationChain( llm=chat_gpt3_5, 
                            #                 memory=ConversationSummaryBufferMemory( #ConversationSummaryBufferMemory(llm=OpenAI(),k=4)
                            #                     llm=OpenAI(),
                            #_memory                     max_token_limit=250),
                            #                 verbose=False,
                            #                 prompt = PLANTILLA
                            #                 ),
                            # ------------------------------------- Buffer Memory -------
                            "chain": ConversationChain( llm=chat_gpt3_5, 
                                            memory=ConversationTokenBufferMemory(#ConversationBufferMemory( #ConversationSummaryBufferMemory(llm=OpenAI(),k=4)
                                                llm=OpenAI(),
                                                # max_history = 6,
                                                max_token_limit = max_token_limit_memory),
                                            verbose=False,
                                            prompt = PLANTILLA
                                            ),
                            
                            # Prompt Token Counter to not exceed the limit
                            "counter_tokens":0,
                          
                            # Input token count to estimate cost. Human
                            "input_tokens":0,
                            
                            # Output token count to estimate cost. Model
                            "output_tokens":0,
                            
                            # Costos TOTALES 
                            "total_inputs_cost":  0,             
                            "total_outputs_cost":  0
        }
        
              
    # print("valor:",dic_memory[id])
    return dic_memory,falla_memoria#dic_memory

def fecha_hora():
    zona_horaria_colombia = pytz.timezone('America/Bogota')
    hora_actual_colombia = datetime.datetime.now(zona_horaria_colombia)

    # Formatea la hora en un formato legible
    fecha_hora_formateada = hora_actual_colombia.strftime('%Y-%m-%d %H:%M:%S')

    # Imprime la hora en Colombia formateada
    print(f"----------------- {fecha_hora_formateada} -----------------")
    return fecha_hora_formateada

def main(falla_memoria=False):
    # try:
        print("Starting bot...")


        # mensajes=[]
        offset = 0
        count = 0
        COSTO_TOTAL = 0
        token_count_memory = 0
        tokens_user = 0
        tokens_ia = 0
        cost_input_model =0.0015/1000 #usd/ 1K tokens gpt-3.5-turbo
        cost_output_model = 0.002/1000 #usd/ 1K tokens gpt-3.5-turbo
        
        max_tokens_limit_user = 187
        max_token_limit_memory = 600
        max_tokens_completion = 360
        offset_prevention = 0
        
        limite_tokens = 4097 - max_tokens_completion  -offset_prevention   #dic_memory[id]['counter_tokens'] gpt-3.5-turbo 4,097 tokens, para que se accione antes de generar error
        print(f"Limite de tokens por prompt: {limite_tokens} tokens")
        
        
        
        dic_memory = {} # {"<id>":[memory, sum_prompt_tokens, cost]}
        df = pd.DataFrame(
            columns=['Id','date','time','username','first_name','last_name','Mensaje','IA_rta'])
        tiempo_ON = fecha_hora() 
        tokens = tokens_plantilla
        chat_gpt3_5 = ChatOpenAI(
            openai_api_key=apikey,
            temperature=0,
            model='gpt-3.5-turbo',#'gpt-4',
            max_tokens=max_tokens_completion,
        )   
                
        while True: 
            print('.')              
            updates = get_updates(offset)
            
            if updates:
                
                tiempo = fecha_hora()
                print(f"Interacción N°: {count}")
                print(f"Conversaciones: {len(dic_memory)}")
                # print(f"Tokens: {tokens} {datetime.datetime.now(pytz.timezone('America/Bogota')).time().strftime('%H:%M:%S')}")
                
                for update in updates:
                    offset = update["update_id"] + 1
                    try:
                        

                        chat_id = str(update["message"]["chat"]['id'])
                        user_message = update["message"]["text"]
                        
                        try:
                            date = update["message"]['date']
                        except: date = "nan"
                        try:
                            username= update["message"]["from"]['username']
                        except: username = "nan"
                        
                        try:
                            first_name = update["message"]["from"]['first_name']
                        except: first_name = "nan"
                        try:
                            last_name = update["message"]["from"]['last_name']
                        except: last_name = "nan" 
                        
                    except:
                        chat_id = str(update["edited_message"]["chat"]['id'] )    
                        user_message = update["edited_message"]["text"]
                        
                        try: date = update["edited_message"]['date']
                        except: date = "nan"
                        
                        try: username= update["edited_message"]["from"]['username']
                        except: username = "nan"
                        
                        try:first_name = update["edited_message"]["from"]['first_name']
                        except: first_name = "nan"
                        
                        try:last_name = update["edited_message"]["from"]['last_name']
                        except: last_name = "nan" 
                        
                    
                    tokens_user = int(len(get_encoding("cl100k_base").encode(user_message)))
                    
                    if tokens_user < max_tokens_limit_user:
                        if chat_id in dic_memory:
                            
                            token_count_memory = dic_memory[chat_id]['input_tokens'] + dic_memory[chat_id]['output_tokens']
                            
                            if token_count_memory>max_token_limit_memory:
                                token_count_memory = max_token_limit_memory
                                                                    # por ahora no considero el numeoro exacto de tokens en memoria memory.chat_memory.get_token_count()
                            dic_memory[chat_id]['counter_tokens'] = tokens_user + tokens_plantilla + token_count_memory # Igual porque es el contador de tokens del prompt
                                                                                                    # el cual utilizo para no exeder el límite
                                                                                                    
                        dic_memory,falla_memoria = almacenar_conversacion(dic_memory, chat_id,               
                                                            chat_gpt3_5,
                                                            tokens_plantilla,   limite_tokens ,max_token_limit_memory,
                                                            falla_memoria
                                                            )
                        dic_memory[chat_id]['counter_tokens'] = tokens_user + tokens_plantilla + token_count_memory
                    else:pass   
                        
                    
                    print(f"User {username} | Received message: {user_message}")
                    # print(dic_memory)
                    # conversacion = dic_memory[chat_id]
                    if (falla_memoria==False) & (tokens_user < max_tokens_limit_user):
                        
                        r = dic_memory[chat_id]['chain'].predict(input=user_message)
                        
                        tokens_ia = int(len(get_encoding("cl100k_base").encode(r)))
                       
                        dic_memory[chat_id]['input_tokens']+=tokens_user
                        dic_memory[chat_id]['output_tokens']+=tokens_ia
                        
                        actual_message_imput_cost =  (tokens_user+tokens_plantilla+token_count_memory)*cost_input_model
                        actual_message_output_cost = tokens_ia*cost_output_model
                        tokens_totales = tokens_user+tokens_plantilla+token_count_memory + tokens_ia
                        dic_memory[chat_id]['total_inputs_cost']+=actual_message_imput_cost
                        dic_memory[chat_id]['total_outputs_cost']+=actual_message_output_cost
                        
                        COSTO_TOTAL+=actual_message_imput_cost+actual_message_output_cost
                        
                        # print(f"Conversaciones Almacenadas: {len(dic_memory)}\n")
                        print(f"\n--------- Tokens y Costos Aproximados | Usuario: {username} ----------\n")
                        print(f"Tokens aprox en memoria: {token_count_memory}")
                        print(f"Tokens totales en buffer: {int(len(get_encoding('cl100k_base').encode(str(dic_memory[chat_id]['chain'].memory.buffer))))}")
                        print("Inputs:")
                        print(f" Costo Input: {round(actual_message_imput_cost,4)} USD, por {dic_memory[chat_id]['counter_tokens']} Tokens") # (tok_template+tok_memory+token_messages) * input_cost
                        print(f" Costo Total Inputs: {round(dic_memory[chat_id]['total_inputs_cost'],4)} USD")
                        print("Outputs:")
                        print(f" Costo Output: {round(actual_message_output_cost,4)} USD por {tokens_ia} Tokens")
                        print(f" Costo Total Output: {round(dic_memory[chat_id]['total_outputs_cost'],4)} USD")
                        print("Acumulado:")
                        print(f"Costo Acumulado del Usuario: {round(dic_memory[chat_id]['total_inputs_cost']+dic_memory[chat_id]['total_outputs_cost'],2)} USD\n")
                        print("-------------------------------------------------------------------------")
                        print(f"         COSTO TOTAL ACUMULADO: {round(COSTO_TOTAL,4)} USD")
                        print("-------------------------------------------------------------------------\n")
                        
                        # print(f"Tokens aproximados en memoria: {dic_memory[chat_id][1]}")
                    elif tokens_user > max_tokens_limit_user:
                        print(f"********** {tiempo}  : Límite de tokens de usuario superado ********")
                        r="Oh, parece que tu mensaje es demasiado extenso.📝 Para ofrecerte la mejor asistencia, sería genial si pudieras resumirlo o hacerme una pregunta más concisa.😊 Estoy aquí para ayudarte 💬"
                        tokens_user = 0
                    elif(falla_memoria==True):
                        print(f"********** {tiempo}  : Límite de tokens superado ********")
                        r="¡Ups! Parece que he tenido un pequeño fallo de memoria, ¡me disculpo por eso! 😅 ¿Puedes recordarme sobre qué estábamos hablando? Estoy aquí para ayudarte en lo que necesites."
                        dic_memory = {}
                        falla_memoria=False
                    
                    print(f"ai: {r}")
                    print('')
                    # if "salir123" in ia_rta.lower():
                    #     break 
                    
                    send_messages(chat_id, r)
                    
                    nuevo_registro = {'Id':str(chat_id),
                                    # 'date':date,
                                    'time':str(tiempo),
                                    'username':str(username),
                                    'first_name':str(first_name),
                                    'last_name':str(last_name),
                                    'Mensaje':str(user_message),
                                    'user_tokens': int(tokens_user),
                                    'IA_rta':str(r),
                                    'ia_tokens': int(tokens_ia),
                                    'memory_tokens':int(token_count_memory)
                                    }
                    
                    # lista_registro = [valor for valor in nuevo_registro.values()]
                    # print(str(tuple(nuevo_registro.values())),tuple(nuevo_registro.values()))
                    # cargar_registro_en_BD(bd="BOT_3.db",registro=tuple(nuevo_registro.values()))
                    
                    df = pd.concat([df,pd.DataFrame(nuevo_registro, index=[count])])
                    count+=1
                    # df, M.append(nuevo_registro,ignore_index=True)
                    if (len(df)>=5) & (len(df)%5==0):
                        aux= tiempo_ON.replace(' ','_').replace(':','').replace('-','_')
                        # aux= aux.replace(':','')
                        # aux= aux.replace('-','_')
                        df.to_excel(f"./hist/historial_completo_{aux}.xlsx")
            else:
                time.sleep(1)
    # except:
    #     main(falla_memoria=True)
        
        
if __name__ == '__main__':
    main()