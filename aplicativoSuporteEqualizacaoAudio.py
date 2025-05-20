# IMPORTAÇÕES 

from librosa import amplitude_to_db as amplitudeToDb, load, piptrack
from librosa.feature import rms
from matplotlib.pyplot import figure, grid, legend, title, vlines, xlabel, xlim, xscale, xticks, ylabel, ylim, yscale, yticks
from numpy import arange, float32, max, min, round
from pandas import DataFrame
from streamlit import columns, container, dataframe, error, fragment, file_uploader as fileUploader, markdown, multiselect, pyplot, radio, selectbox, set_page_config as setPageConfig, sidebar, slider, spinner, stop, tabs, title

# CONFIGURAÇÕES DO FRAMEWORK "STREAMLIT"

setPageConfig(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

# VARIÁVEIS GLOBAIS

centralizacaoQuadro = False 
conversaoMono = True 
decibelMaximoEixoYGrafico = 0 
duracaoMaximaSegundosCarregamentoAudio = None
escolhaTamanhoJanelaSTFT = 1024
frequenciaMaximaAudivel = 20000 
frequenciaMinimaAudivel = 20 
frequenciasDestaqueEscalaLinearGraficoLinhasVerticais = [20, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000] # Definido arbitrariamente
frequenciasDestaqueEscalaLogaritmicaGraficoLinhasVerticais = [20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000] # Definido arbitrariamente
funcaoAgregacaoCanais = None 
inicioSegundosCarregamentoAudio = 0.0 
limiteAbaixoReferenciaConsideradoSilencio = 1e-12 
limiteAbaixoRmsConsideradoSilencio = 1e-03
limiteDecibeisAbaixoPico = 130.0 
limiteMinimoObtencaoDadosFrequencias = 1e-01
listaTamanhosJanelaSTFT = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
modoPreenchimento = None 
taxaAmostragem = None
tipoJanelaSTFT = "hann"
tipoMatrizSaida = float32 
tipoReamostragem = "soxr_hq" 
textosFrequenciasDestaqueEscalaLinearGraficoLinhasVerticais = [20, "1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "11K", "12K", "13K", "14K", "15K", "16K", "17K", "18K", "19K", "20K"] # Definido arbitrariamente
textosFrequenciasDestaqueEscalaLogaritmicaGraficoLinhasVerticais = [20, 30, 40, 50, 60, 70, 80, 90, "1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "15K", "20K"] # Definido arbitrariamente

# FUNÇÕES AUXILIARES

def exibirGraficoFrequenciasDecibeisUteisSerieTemporalAudio(listaFrequenciasUteisSerieTemporal, listaDecibeisUteisSerieTemporal, escalaEixoFrequencias):

    graficoFrequenciasDecibeisUteisSerieTemporalAudio = figure(figsize=(20,7))

    ylabel(ylabel="Decibéis")
    ylim(0, decibelMaximoEixoYGrafico)
    yscale("linear")
    yticks(ticks=arange(start=0, stop=decibelMaximoEixoYGrafico, step=5), labels=arange(start=0, stop=decibelMaximoEixoYGrafico, step=5))
    xlabel(xlabel="Frequências")
    xscale(value=escalaEixoFrequencias)

    if(escalaEixoFrequencias == "linear"):
        xticks(ticks=frequenciasDestaqueEscalaLinearGraficoLinhasVerticais, labels=textosFrequenciasDestaqueEscalaLinearGraficoLinhasVerticais, rotation=45)
        xlim((frequenciaMinimaAudivel-100), (frequenciaMaximaAudivel+100))
    else:
        xticks(ticks=frequenciasDestaqueEscalaLogaritmicaGraficoLinhasVerticais, labels=textosFrequenciasDestaqueEscalaLogaritmicaGraficoLinhasVerticais, rotation=45)
        xlim((frequenciaMinimaAudivel-1), (frequenciaMaximaAudivel+1000))

    vlines(x=listaFrequenciasUteisSerieTemporal, ymin=0, ymax=listaDecibeisUteisSerieTemporal, colors="mediumblue", linestyles="solid", linewidths=1)
    grid(axis="y")
    
    pyplot(fig=graficoFrequenciasDecibeisUteisSerieTemporalAudio, clear_figure=False, use_container_width=True)

def exibirGraficosSobrepostosFrequenciasDecibeisUteisSeriesTemporaisAudios(listaFrequenciasUteisSerieTemporalAudioOriginal, listaDecibeisUteisSerieTemporalAudioOriginal,
                                                                           listaFrequenciasUteisSerieTemporalAudioFinal, listaDecibeisUteisSerieTemporalAudioFinal,
                                                                           escalaEixoFrequencias):

    graficosSobrepostos = figure(figsize=(20,3.5))

    ylabel(ylabel="Decibéis")
    ylim(0, decibelMaximoEixoYGrafico)
    yscale("linear")
    yticks(ticks=arange(start=0, stop=decibelMaximoEixoYGrafico, step=5), labels=arange(start=0, stop=decibelMaximoEixoYGrafico, step=5))
    xlabel(xlabel="Frequências")
    xscale(value=escalaEixoFrequencias)

    if(escalaEixoFrequencias == "linear"):
        xticks(ticks=frequenciasDestaqueEscalaLinearGraficoLinhasVerticais, labels=textosFrequenciasDestaqueEscalaLinearGraficoLinhasVerticais, rotation=45)
        xlim((frequenciaMinimaAudivel-100), (frequenciaMaximaAudivel+100))
    else:
        xticks(ticks=frequenciasDestaqueEscalaLogaritmicaGraficoLinhasVerticais, labels=textosFrequenciasDestaqueEscalaLogaritmicaGraficoLinhasVerticais, rotation=45)
        xlim((frequenciaMinimaAudivel-1), (frequenciaMaximaAudivel+1000))

    grid(axis="y")
    vlines(x=listaFrequenciasUteisSerieTemporalAudioOriginal, ymin=0, ymax=listaDecibeisUteisSerieTemporalAudioOriginal, colors="gray", linestyles="solid", label="Áudio Original", linewidths=1)
    vlines(x=listaFrequenciasUteisSerieTemporalAudioFinal, ymin=0, ymax=listaDecibeisUteisSerieTemporalAudioFinal, colors="mediumblue", linestyles="solid", label="Áudio Final", linewidths=1)
    legend(bbox_to_anchor=(0.5, 1.127))

    pyplot(fig=graficosSobrepostos, clear_figure=False, use_container_width=True)

def filtrarFrequenciasAmplitudesUteisSerieTemporalAudio(frequenciasInstantaneasSerieTemporal, amplitudesFrequenciasInstantaneasSerieTemporal, 
                                                        posicoesLinhasNaoNulasFrequenciasAmplitudesSerieTemporal,
                                                        posicoesColunasNaoNulasFrequenciasAmplitudesSerieTemporal):

    listaFrequenciasUteis = []
    listaAmplitudesUteis = []

    for posicaoLinhaColuna in range(0, posicoesLinhasNaoNulasFrequenciasAmplitudesSerieTemporal.shape[0]):
        listaFrequenciasUteis.append(frequenciasInstantaneasSerieTemporal[posicoesLinhasNaoNulasFrequenciasAmplitudesSerieTemporal[posicaoLinhaColuna]][posicoesColunasNaoNulasFrequenciasAmplitudesSerieTemporal[posicaoLinhaColuna]])
        listaAmplitudesUteis.append(amplitudesFrequenciasInstantaneasSerieTemporal[posicoesLinhasNaoNulasFrequenciasAmplitudesSerieTemporal[posicaoLinhaColuna]][posicoesColunasNaoNulasFrequenciasAmplitudesSerieTemporal[posicaoLinhaColuna]])

    return listaFrequenciasUteis, listaAmplitudesUteis

def importarAudio(chaveExclusiva):

    audio = fileUploader(label="Importação de Áudio", type=["flac", "mat", "mp3", "mp4", "ogg", "wav", "wma"], accept_multiple_files=False, key=chaveExclusiva, help=None, 
                         on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed")

    return audio

def obterFrequenciasAmplitudesSerieTemporalAudio(serieTemporalAudio, taxaAmostragemAudio, tamanhoJanelaSTFT, amplitudeReferencia):
    
    frequenciasInstantaneas, amplitudesFrequenciasInstantaneas = piptrack(y=serieTemporalAudio, sr=taxaAmostragemAudio, S=None, n_fft=tamanhoJanelaSTFT, 
                                                                          hop_length=tamanhoJanelaSTFT//4, fmin=frequenciaMinimaAudivel, 
                                                                          fmax=(frequenciaMaximaAudivel + 1), threshold=limiteMinimoObtencaoDadosFrequencias, 
                                                                          win_length=tamanhoJanelaSTFT, window=tipoJanelaSTFT, center=centralizacaoQuadro, 
                                                                          pad_mode=modoPreenchimento, ref=amplitudeReferencia)
    
    return frequenciasInstantaneas, amplitudesFrequenciasInstantaneas

@fragment
def obterFrequenciasEspecificasSelecionadasExibicaoDecibeis(listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal, listaFrequenciasUteisArredondadasSerieTemporalAudioFinal,
                                                            listaDecibeisUteisArredondadosSerieTemporalAudioOriginal, listaDecibeisUteisArredondadosSerieTemporalAudioFinal,
                                                            listaFrequenciasComuns, menorDiferencaDecibeis):
    
    frequenciasEspecificasSelecionadas = multiselect(label="Seleção de Frequências", options=sorted(listaFrequenciasComuns), 
                                                     default=None, key=None, help=None, on_change=None, max_selections=None, placeholder="Selecione as frequências desejadas", disabled=False, 
                                                     label_visibility="collapsed")

    listaOrdenadaFrequenciasEspecificasSelecionadas = sorted(frequenciasEspecificasSelecionadas)

    # Filtragem dos decibéis de frequências úteis específicas das séries temporais dos áudios original e final
    listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal = []
    listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal = []

    listaSugestoesDecibeis = []

    contador = 0
    
    for frequenciaEspecifica in listaOrdenadaFrequenciasEspecificasSelecionadas:
        posicaoFrequenciaEspecifica = listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal.index(frequenciaEspecifica)
        listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal.append(listaDecibeisUteisArredondadosSerieTemporalAudioOriginal[posicaoFrequenciaEspecifica])

        posicaoFrequenciaEspecifica = listaFrequenciasUteisArredondadasSerieTemporalAudioFinal.index(frequenciaEspecifica)
        listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal.append(listaDecibeisUteisArredondadosSerieTemporalAudioFinal[posicaoFrequenciaEspecifica])

        if(listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal[contador] > listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal[contador]): 
            texto = "Aumentar " + str(listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal[contador] - listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal[contador] - menorDiferencaDecibeis) + " dB"

            listaSugestoesDecibeis.append(texto)
        elif(listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal[contador] < listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal[contador]):
            texto = "Diminuir " + str(listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal[contador] - listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal[contador] - menorDiferencaDecibeis) + " dB"

            listaSugestoesDecibeis.append(texto)            
        else:
            listaSugestoesDecibeis.append("Manter")

        contador += 1

    # Sugestão de equalização para as frequências especificadas
    listaOrdenadaFrequenciasEspecificasSelecionadas = [str(valor) for valor in listaOrdenadaFrequenciasEspecificasSelecionadas]
    listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal = [str(valor) for valor in listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal]
    listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal = [str(valor) for valor in listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal]

    dataFrameResultado = DataFrame(
        {
            "Frequências": listaOrdenadaFrequenciasEspecificasSelecionadas,
            "Decibéis Originais": listaDecibeisFrequenciasEspecificasSerieTemporalAudioOriginal,
            "Decibéis Finais": listaDecibeisFrequenciasEspecificasSerieTemporalAudioFinal,
            "Sugestões de Equalização": listaSugestoesDecibeis,
        }
    )

    dataframe(dataFrameResultado, use_container_width=True , hide_index=True)

def obterRmsQuadrosSerieTemporalAudio(serieTemporalAudio, tamanhoJanelaSTFT):
    
    rmsQuadrosAudio = rms(y=serieTemporalAudio, S=None, frame_length=tamanhoJanelaSTFT, hop_length=tamanhoJanelaSTFT//4, center=centralizacaoQuadro, 
                          pad_mode=modoPreenchimento, dtype=tipoMatrizSaida)
    
    return rmsQuadrosAudio

def obterSerieTemporalTaxaAmostragemAudio(audio):
    
    serieTemporalAudio, taxaAmostragemAudio = load(path=audio, sr=taxaAmostragem, mono=conversaoMono, offset=inicioSegundosCarregamentoAudio, 
                                                   duration=duracaoMaximaSegundosCarregamentoAudio, dtype=tipoMatrizSaida, res_type=tipoReamostragem)
    
    return serieTemporalAudio, taxaAmostragemAudio

def verificarUsoTamanhoStftEscolhido(serieTemporalAudioOriginal, serieTemporalAudioFinal, escolhaTamanhoJanelaSTFT):
    
    if((escolhaTamanhoJanelaSTFT > serieTemporalAudioOriginal.shape[0]) or (escolhaTamanhoJanelaSTFT > serieTemporalAudioFinal.shape[0])):
        error("ERRO: O TAMANHO DA JANELA DE ANÁLISE DEVE SER MENOR QUE {}!\n".format(min([serieTemporalAudioOriginal.shape[0], serieTemporalAudioFinal.shape[0]])), icon=None)

        stop()

# EXECUÇÃO PRINCIPAL

if __name__ == '__main__':
    # Importações dos áudios original e final
    coluna1ImportacaoAudio, coluna2ImportacaoAudio = columns(spec=2, gap="large")    
    
    with coluna1ImportacaoAudio:
        with spinner(text="Carregando campo..."):
            markdown("**IMPORTAÇÃO DO ÁUDIO ORIGINAL:**")
            audioOriginal = importarAudio(1)

    with coluna2ImportacaoAudio:
        with spinner(text="Carregando campo..."):
            markdown("**IMPORTAÇÃO DO ÁUDIO FINAL:**")
            audioFinal = importarAudio(2)

    # Ações após obtenção dos áudios original e final
    if((audioOriginal is not None) and (audioFinal is not None)):
        # Obtenção de séries temporais e taxas de amostragens dos áudios original e final
        serieTemporalAudioOriginal, taxaAmostragemAudioOriginal = obterSerieTemporalTaxaAmostragemAudio(audioOriginal)
        serieTemporalAudioFinal, taxaAmostragemAudioFinal = obterSerieTemporalTaxaAmostragemAudio(audioFinal)
        
        # Obtenção do tamanho de janela STFT ideal
        indiceTamanhoJanelaSTFT = -1

        for tamanho in listaTamanhosJanelaSTFT:
            if((serieTemporalAudioOriginal.shape[0] > tamanho) and (serieTemporalAudioFinal.shape[0] > tamanho)):
                indiceTamanhoJanelaSTFT += 1

        # Definição de menu lateral
        with sidebar:
            title("Configurações")

            escolhaTamanhoJanelaSTFT = selectbox("Tamanho da Janela de Análise:", listaTamanhosJanelaSTFT, indiceTamanhoJanelaSTFT, placeholder="Escolha uma opção")
            escolhaEscalaEixoFrequencias = radio("Escala do Eixo de Frequências:", ["linear", "logaritmica"], 0, horizontal=True)

        dicionarioEscalasEixos = {
            "linear": "linear",
            "logaritmica": "log"
        }
        
        # Verificação de possibilidade de uso do tamanho STFT escolhido
        verificarUsoTamanhoStftEscolhido(serieTemporalAudioOriginal, serieTemporalAudioFinal, escolhaTamanhoJanelaSTFT)
        
        # Obtenção de RMS (Raiz Quadrada Média) dos quadros das séries temporais dos áudios original e final
        rmsQuadrosSerieTemporalAudioOriginal = obterRmsQuadrosSerieTemporalAudio(serieTemporalAudioOriginal, escolhaTamanhoJanelaSTFT)
        rmsQuadrosSerieTemporalAudioFinal = obterRmsQuadrosSerieTemporalAudio(serieTemporalAudioFinal, escolhaTamanhoJanelaSTFT)

        # Verificação de RMS válido para os áudios original e final
        if((max(rmsQuadrosSerieTemporalAudioOriginal) < limiteAbaixoRmsConsideradoSilencio) and (max(rmsQuadrosSerieTemporalAudioFinal) < limiteAbaixoRmsConsideradoSilencio)):
            error("ERRO: ÁUDIOS SILENCIOSOS!\n", icon=None)

            stop()

        elif(max(rmsQuadrosSerieTemporalAudioOriginal) < limiteAbaixoRmsConsideradoSilencio):
            error("ERRO: ÁUDIO ORIGINAL SILENCIOSO!\n", icon=None)

            stop()
        
        elif(max(rmsQuadrosSerieTemporalAudioFinal) < limiteAbaixoRmsConsideradoSilencio):
            error("ERRO: ÁUDIO FINAL SILENCIOSO!\n", icon=None)

            stop()

        try:
            # Limitação de duração máxima das séries temporais dos áudios original e final
            if(serieTemporalAudioOriginal.shape[0] > escolhaTamanhoJanelaSTFT):
                serieTemporalAudioOriginal = serieTemporalAudioOriginal[0:escolhaTamanhoJanelaSTFT]

            if(serieTemporalAudioFinal.shape[0] > escolhaTamanhoJanelaSTFT):
                serieTemporalAudioFinal = serieTemporalAudioFinal[0:escolhaTamanhoJanelaSTFT] 

            # Obtenção de frequências instantâneas e respectivas amplitudes das séries temporais dos áudios original e final
            frequenciasInstantaneasSerieTemporalAudioOriginal, amplitudesFrequenciasInstantaneasSerieTemporalAudioOriginal = obterFrequenciasAmplitudesSerieTemporalAudio(serieTemporalAudioOriginal,
                                                                                                                                                                        taxaAmostragemAudioOriginal,
                                                                                                                                                                        escolhaTamanhoJanelaSTFT,
                                                                                                                                                                        rmsQuadrosSerieTemporalAudioOriginal.min())
            
            frequenciasInstantaneasSerieTemporalAudioFinal, amplitudesFrequenciasInstantaneasSerieTemporalAudioFinal = obterFrequenciasAmplitudesSerieTemporalAudio(serieTemporalAudioFinal,
                                                                                                                                                                    taxaAmostragemAudioFinal,
                                                                                                                                                                    escolhaTamanhoJanelaSTFT,
                                                                                                                                                                    rmsQuadrosSerieTemporalAudioFinal.min())

            # Filtragem de frequências e amplitudes úteis das séries temporais dos áudios original e final
            posicoesLinhasNaoNulasSerieTemporalAudioOriginal, posicoesColunasNaoNulasSerieTemporalAudioOriginal = frequenciasInstantaneasSerieTemporalAudioOriginal.nonzero()
            posicoesLinhasNaoNulasSerieTemporalAudioFinal, posicoesColunasNaoNulasSerieTemporalAudioFinal = frequenciasInstantaneasSerieTemporalAudioFinal.nonzero()

            listaFrequenciasUteisSerieTemporalAudioOriginal, listaAmplitudesUteisSerieTemporalAudioOriginal = filtrarFrequenciasAmplitudesUteisSerieTemporalAudio(frequenciasInstantaneasSerieTemporalAudioOriginal,
                                                                                                                                                                amplitudesFrequenciasInstantaneasSerieTemporalAudioOriginal,
                                                                                                                                                                posicoesLinhasNaoNulasSerieTemporalAudioOriginal,
                                                                                                                                                                posicoesColunasNaoNulasSerieTemporalAudioOriginal)
            
            listaFrequenciasUteisSerieTemporalAudioFinal, listaAmplitudesUteisSerieTemporalAudioFinal = filtrarFrequenciasAmplitudesUteisSerieTemporalAudio(frequenciasInstantaneasSerieTemporalAudioFinal,
                                                                                                                                                            amplitudesFrequenciasInstantaneasSerieTemporalAudioFinal,
                                                                                                                                                            posicoesLinhasNaoNulasSerieTemporalAudioFinal,
                                                                                                                                                            posicoesColunasNaoNulasSerieTemporalAudioFinal)        
            
            # Arredondamento e conversão em inteiro das frequências úteis das séries temporais dos áudios original e final
            listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal = [int(round(frequencia, 2)) for frequencia in listaFrequenciasUteisSerieTemporalAudioOriginal]
            listaFrequenciasUteisArredondadasSerieTemporalAudioFinal = [int(round(frequencia, 2)) for frequencia in listaFrequenciasUteisSerieTemporalAudioFinal]
        
            # Conversão, em decibéis, de amplitudes úteis de frequências instantâneas das séries temporais dos áudios original e final
            listaDecibeisUteisSerieTemporalAudioOriginal = amplitudeToDb(S=listaAmplitudesUteisSerieTemporalAudioOriginal, ref=limiteAbaixoReferenciaConsideradoSilencio, amin=limiteMinimoObtencaoDadosFrequencias, top_db=limiteDecibeisAbaixoPico).tolist()
            listaDecibeisUteisSerieTemporalAudioFinal = amplitudeToDb(S=listaAmplitudesUteisSerieTemporalAudioFinal, ref=limiteAbaixoReferenciaConsideradoSilencio, amin=limiteMinimoObtencaoDadosFrequencias, top_db=limiteDecibeisAbaixoPico).tolist()
            
            # Arredondamento e conversão em inteiro dos decibéis das frequências úteis das séries temporais dos áudios original e final
            listaDecibeisUteisArredondadosSerieTemporalAudioOriginal = [int(round(decibel, 0)) for decibel in listaDecibeisUteisSerieTemporalAudioOriginal]
            listaDecibeisUteisArredondadosSerieTemporalAudioFinal = [int(round(decibel, 0)) for decibel in listaDecibeisUteisSerieTemporalAudioFinal]

            # Obtenção do maior decibel útel entre as listas de decibéis das séries temporais dos áudios original e final
            decibelMaximoSerieTemporalOriginal = max(listaDecibeisUteisArredondadosSerieTemporalAudioOriginal)
            decibelMaximoSerieTemporalFinal = max(listaDecibeisUteisArredondadosSerieTemporalAudioFinal) 
            
            decibelMaximoEixoYGrafico = (max([decibelMaximoSerieTemporalOriginal, decibelMaximoSerieTemporalFinal]) + 5)

            # Exibição de gráficos (sobrepostos e separados) de linhas verticais contendo frequências e decibéis úteis das séries temporais dos áudios original e final
            abaExibicaoGraficosSeparados, abaExibicaoGraficosSobrepostos = tabs(tabs=["GRÁFICOS SEPARADOS", "GRÁFICOS SOBREPOSTOS"])

            with abaExibicaoGraficosSeparados:
                coluna1ExibicaoGrafico, coluna2ExibicaoGrafico = columns(spec=2, gap="large")

                with coluna1ExibicaoGrafico:
                    with spinner(text="Carregando gráfico..."):
                        exibirGraficoFrequenciasDecibeisUteisSerieTemporalAudio(listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal, listaDecibeisUteisArredondadosSerieTemporalAudioOriginal, dicionarioEscalasEixos[escolhaEscalaEixoFrequencias])

                with coluna2ExibicaoGrafico:
                    with spinner(text="Carregando gráfico..."):
                        exibirGraficoFrequenciasDecibeisUteisSerieTemporalAudio(listaFrequenciasUteisArredondadasSerieTemporalAudioFinal, listaDecibeisUteisArredondadosSerieTemporalAudioFinal, dicionarioEscalasEixos[escolhaEscalaEixoFrequencias])

            with abaExibicaoGraficosSobrepostos:
                with spinner(text="Carregando gráfico..."):
                    exibirGraficosSobrepostosFrequenciasDecibeisUteisSeriesTemporaisAudios(listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal, listaDecibeisUteisArredondadosSerieTemporalAudioOriginal,
                                                                                            listaFrequenciasUteisArredondadasSerieTemporalAudioFinal, listaDecibeisUteisArredondadosSerieTemporalAudioFinal,
                                                                                            dicionarioEscalasEixos[escolhaEscalaEixoFrequencias])

            listaFrequenciasComuns = list(set(listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal) & set(listaFrequenciasUteisArredondadasSerieTemporalAudioFinal))

            # Verificação da necessidade de aumento ou diminuição de volume
            menorDiferencaDecibeis = 0

            if(len(listaFrequenciasComuns) == 0):
                with container(height=None, border=False):
                    with spinner(text="Carregando campo..."):
                        markdown("**AJUSTE DE VOLUME:** Manter")

            elif(max(listaDecibeisUteisArredondadosSerieTemporalAudioOriginal) > max(listaDecibeisUteisArredondadosSerieTemporalAudioFinal)):
                menorDiferencaDecibeis = max(listaDecibeisUteisArredondadosSerieTemporalAudioOriginal)

                for frequencia in listaFrequenciasComuns:
                    posicaoFrequenciaOriginal = listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal.index(frequencia)
                    posicaoFrequenciaFinal = listaFrequenciasUteisArredondadasSerieTemporalAudioFinal.index(frequencia)

                    if((listaDecibeisUteisArredondadosSerieTemporalAudioOriginal[posicaoFrequenciaOriginal] - listaDecibeisUteisArredondadosSerieTemporalAudioFinal[posicaoFrequenciaFinal]) < menorDiferencaDecibeis):
                        menorDiferencaDecibeis = (listaDecibeisUteisArredondadosSerieTemporalAudioOriginal[posicaoFrequenciaOriginal] - listaDecibeisUteisArredondadosSerieTemporalAudioFinal[posicaoFrequenciaFinal])

                        if(menorDiferencaDecibeis <= 0):
                            break

                with container(height=None, border=False):
                    with spinner(text="Carregando campo..."):
                        if(menorDiferencaDecibeis <= 0):
                            markdown("**AJUSTE DE VOLUME:** Manter")
                        else:
                            markdown(f"**AJUSTE DE VOLUME:** Aumentar {str(menorDiferencaDecibeis)} dB")

            else:
                menorDiferencaDecibeis = max(listaDecibeisUteisArredondadosSerieTemporalAudioFinal)

                for frequencia in listaFrequenciasComuns:
                    posicaoFrequenciaFinal = listaFrequenciasUteisArredondadasSerieTemporalAudioFinal.index(frequencia)
                    posicaoFrequenciaOriginal = listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal.index(frequencia)

                    if((listaDecibeisUteisArredondadosSerieTemporalAudioFinal[posicaoFrequenciaFinal] - listaDecibeisUteisArredondadosSerieTemporalAudioOriginal[posicaoFrequenciaOriginal]) < menorDiferencaDecibeis):
                        menorDiferencaDecibeis = (listaDecibeisUteisArredondadosSerieTemporalAudioFinal[posicaoFrequenciaFinal] - listaDecibeisUteisArredondadosSerieTemporalAudioOriginal[posicaoFrequenciaOriginal])

                        if(menorDiferencaDecibeis <= 0):
                            break

                with container(height=None, border=False):
                    with spinner(text="Carregando campo..."):
                        if(menorDiferencaDecibeis <= 0):
                            markdown("**AJUSTE DE VOLUME:** Manter")
                        else:
                            markdown(f"**AJUSTE DE VOLUME:** Diminuir {str(menorDiferencaDecibeis)} dB")

            # Obtenção de frequências específicas selecionadas para exibição de respectivos decibéis
            with container(height=None, border=False):
                with spinner(text="Carregando campo..."):
                    markdown("**ESCOLHA DE FREQUÊNCIAS PARA AJUSTE DE DECIBÉIS:**")
                    obterFrequenciasEspecificasSelecionadasExibicaoDecibeis(listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal, listaFrequenciasUteisArredondadasSerieTemporalAudioFinal,
                                                                            listaDecibeisUteisArredondadosSerieTemporalAudioOriginal, listaDecibeisUteisArredondadosSerieTemporalAudioFinal,
                                                                            listaFrequenciasComuns, menorDiferencaDecibeis if menorDiferencaDecibeis > 0 else 0)
        except:
            error("ERRO: TENTE NOVAMENTE COM UM TAMANHO MENOR DE JANELA DE ANÁLISE!\n", icon=None)

            stop()