# IMPORTAÇÕES 

from librosa import amplitude_to_db as amplitudeToDb, load, piptrack
from librosa.effects import trim
from librosa.feature import rms
from matplotlib.pyplot import figure, grid, legend, title, vlines, xlabel, xlim, xscale, xticks, ylabel, ylim, yscale, yticks
from numpy import arange, float32, max, round
from pandas import DataFrame
from streamlit import columns, container, dataframe, error, fragment, file_uploader as fileUploader, markdown, multiselect, pyplot, radio, selectbox, set_page_config as setPageConfig, sidebar, slider, spinner, stop, tabs, title

# CONFIGURAÇÕES DO FRAMEWORK "STREAMLIT"

setPageConfig(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

# VARIÁVEIS GLOBAIS

centralizacaoQuadro = False 
conversaoMono = True 
decibelMaximoEixoYGrafico = 0 
duracaoMaximaSegundosCarregamentoAudio = 5 
duracaoMinimaSegundosSerieTemporalAudio = 1
frequenciaMaximaAudivel = 20000 
frequenciaMinimaAudivel = 20 
frequenciasDestaqueEscalaLinearGraficoLinhasVerticais = [20, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000] # Definido arbitrariamente
frequenciasDestaqueEscalaLogaritmicaGraficoLinhasVerticais = [20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000] # Definido arbitrariamente
funcaoAgregacaoCanais = None 
inicioSegundosCarregamentoAudio = 0.0 
limiteAbaixoReferenciaConsideradoSilencio = 1e-12 
limiteDecibeisAbaixoPico = 130.0 
limiteMinimoObtencaoDadosFrequencias = 1e-01 
modoPreenchimento = None 
taxaAmostragem = None 
tipoMatrizSaida = float32 
tipoReamostragem = "soxr_hq" 
textosFrequenciasDestaqueEscalaLinearGraficoLinhasVerticais = [20, "1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K", "15K", "20K"] # Definido arbitrariamente
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

def obterFrequenciasAmplitudesSerieTemporalAudio(serieTemporalAudio, taxaAmostragemAudio, tamanhoJanelaSTFT, tamanhoSaltoSTFT, tipoJanelaSTFT, amplitudeReferencia):
    
    frequenciasInstantaneas, amplitudesFrequenciasInstantaneas = piptrack(y=serieTemporalAudio, sr=taxaAmostragemAudio, S=None, n_fft=tamanhoJanelaSTFT, 
                                                                          hop_length=tamanhoSaltoSTFT, fmin=frequenciaMinimaAudivel, 
                                                                          fmax=(frequenciaMaximaAudivel + 1), threshold=limiteMinimoObtencaoDadosFrequencias, 
                                                                          win_length=tamanhoJanelaSTFT, window=tipoJanelaSTFT,  center=centralizacaoQuadro, 
                                                                          pad_mode=modoPreenchimento, ref=amplitudeReferencia)
    
    return frequenciasInstantaneas, amplitudesFrequenciasInstantaneas

@fragment
def obterFrequenciasEspecificasSelecionadasExibicaoDecibeis(listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal, listaFrequenciasUteisArredondadasSerieTemporalAudioFinal,
                                                            listaDecibeisUteisArredondadosSerieTemporalAudioOriginal, listaDecibeisUteisArredondadosSerieTemporalAudioFinal,
                                                            listaFrequenciasComuns, menorDiferencaDecibeis):
    
    frequenciasEspecificasSelecionadas = multiselect(label="Seleção de Frequências", options=listaFrequenciasComuns, 
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

def obterRmsQuadrosSerieTemporalAudio(serieTemporalAudio, tamanhoJanelaSTFT, tamanhoSaltoSTFT):
    
    rmsQuadrosAudio = rms(y=serieTemporalAudio, S=None, frame_length=tamanhoJanelaSTFT, hop_length=tamanhoSaltoSTFT, center=centralizacaoQuadro, 
        pad_mode=modoPreenchimento, dtype=tipoMatrizSaida)
    
    return rmsQuadrosAudio

def obterSerieTemporalTaxaAmostragemAudio(audio):
    
    serieTemporalAudio, taxaAmostragemAudio = load(path=audio, sr=taxaAmostragem, mono=conversaoMono, offset=inicioSegundosCarregamentoAudio, 
                                                   duration=duracaoMaximaSegundosCarregamentoAudio, dtype=tipoMatrizSaida, res_type=tipoReamostragem)
    
    return serieTemporalAudio, taxaAmostragemAudio

def retirarSilenciosInicialFinalSerieTemporalAudio(serieTemporalAudio, amplitudeReferencia, tamanhoJanelaSTFT, tamanhoSaltoSTFT):
    
    serieTemporalAparadaAudio, intervaloNaoSilenciosoSerieTemporalAparadaAudio = trim(y=serieTemporalAudio, top_db=limiteAbaixoReferenciaConsideradoSilencio, 
                                                                                       ref=amplitudeReferencia, frame_length=tamanhoJanelaSTFT, 
                                                                                       hop_length=tamanhoSaltoSTFT, aggregate=funcaoAgregacaoCanais)

    return serieTemporalAparadaAudio, intervaloNaoSilenciosoSerieTemporalAparadaAudio

# EXECUÇÃO PRINCIPAL

if __name__ == '__main__':
    
    # Definição de menu lateral
    with sidebar:
        title("Configurações")

        escolhaTempoMaximoAnaliseAudio = slider("Tempo Máximo de Análise do Áudio (Segundos):", 1, duracaoMaximaSegundosCarregamentoAudio, 1, 1)

        escolhaTipoJanelaSTFT = selectbox("Tipo de Janela STFT:", ["Bartlett", "Blackman", "Bohman", "Exponencial", "Gaussiana", "Hamming", "Hann", "Kaiser", "Lanczos", "Parzen", "Retangular", "Taylor", "Triangular", "Tukey"], 6, placeholder="Escolha uma opção")
        escolhaTamanhoJanelaSTFT = selectbox("Tamanho da Janela STFT:", [1024, 2048, 4096, 8192, 16384, 32768], 5, placeholder="Escolha uma opção")
        escolhaPorcentagemSaltoSTFT = selectbox("Porcentagem do Salto STFT:", [0.25, 0.50, 0.75], 0, placeholder="Escolha uma opção")
        
        escolhaEscalaEixoFrequencias = radio("Escala do Eixo de Frequências:", ["linear", "logaritmica"], 1, horizontal=True)

    dicionarioTiposJanelasSTFT = {
        "Bartlett": "bartlett", 
        "Blackman": "blackman", 
        "Bohman": "bohman", 
        "Exponencial": "exponential", 
        "Gaussiana": "gaussian", 
        "Hamming": "hamming", 
        "Hann": "hann", 
        "Kaiser": "kaiser", 
        "Lanczos": "lanczos", 
        "Parzen": "parzen", 
        "Retangular": "boxcar", 
        "Taylor": "taylor", 
        "Triangular": "triang", 
        "Tukey": "tukey" 
    }

    dicionarioPorcentagensSaltosSTFT = {
        0.25: int(escolhaTamanhoJanelaSTFT * 0.25), 
        0.50: int(escolhaTamanhoJanelaSTFT * 0.5), 
        0.75: int(escolhaTamanhoJanelaSTFT * 0.75)
    }

    dicionarioEscalasEixos = {
        "linear": "linear",
        "logaritmica": "log"
    }

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

        # Obtenção de RMS (Raiz Quadrada Média) dos quadros das séries temporais dos áudios original e final
        rmsQuadrosSerieTemporalAudioOriginal = obterRmsQuadrosSerieTemporalAudio(serieTemporalAudioOriginal, escolhaTamanhoJanelaSTFT, dicionarioPorcentagensSaltosSTFT[escolhaPorcentagemSaltoSTFT])
        rmsQuadrosSerieTemporalAudioFinal = obterRmsQuadrosSerieTemporalAudio(serieTemporalAudioFinal, escolhaTamanhoJanelaSTFT, dicionarioPorcentagensSaltosSTFT[escolhaPorcentagemSaltoSTFT])

        # Verificação de RMS válido para os áudios original e final
        if((rmsQuadrosSerieTemporalAudioOriginal.sum() <= 0) and (rmsQuadrosSerieTemporalAudioFinal.sum() <= 0)):
            error("ERRO: ÁUDIOS SILENCIOSOS!\n", icon=None)

            stop()

        elif(rmsQuadrosSerieTemporalAudioOriginal.sum() <= 0):
            error("ERRO: ÁUDIO ORIGINAL SILENCIOSO!\n", icon=None)

            stop()
        
        elif(rmsQuadrosSerieTemporalAudioFinal.sum() <= 0):
            error("ERRO: ÁUDIO FINAL SILENCIOSO!\n", icon=None)

            stop()

        # Retirada de silêncios iniciais e finais das séries temporais dos áudios original e final
        # O silêncio foi associado com o RMS (Raiz Quadrada Média) mínimo dos áudios original e final
        if((serieTemporalAudioOriginal[0] == 0) or (serieTemporalAudioOriginal[serieTemporalAudioOriginal.shape[0] - 1] == 0)): 
            serieTemporalAudioOriginal, intervaloNaoSilenciosoSerieTemporalAudioOriginal = retirarSilenciosInicialFinalSerieTemporalAudio(serieTemporalAudioOriginal, rmsQuadrosSerieTemporalAudioOriginal.min(), escolhaTamanhoJanelaSTFT, dicionarioPorcentagensSaltosSTFT[escolhaPorcentagemSaltoSTFT])
        
        if((serieTemporalAudioFinal[0] == 0) or (serieTemporalAudioFinal[serieTemporalAudioFinal.shape[0] - 1] == 0)): 
            serieTemporalAudioFinal, intervaloNaoSilenciosoSerieTemporalAudioFinal = retirarSilenciosInicialFinalSerieTemporalAudio(serieTemporalAudioFinal, rmsQuadrosSerieTemporalAudioFinal.min(), escolhaTamanhoJanelaSTFT, dicionarioPorcentagensSaltosSTFT[escolhaPorcentagemSaltoSTFT])
        
        # Verificação de duração mínima das séries temporais dos áudios original e final
        duracaoSegundosSerieTemporalAudioOriginal = (serieTemporalAudioOriginal.shape[0] / taxaAmostragemAudioOriginal)
        duracaoSegundosSerieTemporalAudioFinal = (serieTemporalAudioFinal.shape[0] / taxaAmostragemAudioFinal)

        if(duracaoSegundosSerieTemporalAudioOriginal < duracaoMinimaSegundosSerieTemporalAudio):
            error("ERRO: DURAÇÃO DE SÉRIE TEMPORAL ORIGINAL MENOR QUE {} SEGUNDO!\n".format(duracaoMinimaSegundosSerieTemporalAudio), icon=None)

            stop()

        if(duracaoSegundosSerieTemporalAudioFinal < duracaoMinimaSegundosSerieTemporalAudio):
            error("ERRO: DURAÇÃO DE SÉRIE TEMPORAL FINAL MENOR QUE {} SEGUNDO!\n".format(duracaoMinimaSegundosSerieTemporalAudio), icon=None)

            stop()

        # Limitação de duração máxima das séries temporais dos áudios original e final
        if(duracaoSegundosSerieTemporalAudioOriginal > escolhaTempoMaximoAnaliseAudio):
            serieTemporalAudioOriginal = serieTemporalAudioOriginal[0:(taxaAmostragemAudioOriginal * escolhaTempoMaximoAnaliseAudio)]

        if(duracaoSegundosSerieTemporalAudioFinal > escolhaTempoMaximoAnaliseAudio):
            serieTemporalAudioFinal = serieTemporalAudioFinal[0:(taxaAmostragemAudioFinal * escolhaTempoMaximoAnaliseAudio)]            

        # Obtenção de frequências instantâneas e respectivas amplitudes das séries temporais dos áudios original e final
        frequenciasInstantaneasSerieTemporalAudioOriginal, amplitudesFrequenciasInstantaneasSerieTemporalAudioOriginal = obterFrequenciasAmplitudesSerieTemporalAudio(serieTemporalAudioOriginal,
                                                                                                                                                                    taxaAmostragemAudioOriginal,
                                                                                                                                                                    escolhaTamanhoJanelaSTFT,
                                                                                                                                                                    dicionarioPorcentagensSaltosSTFT[escolhaPorcentagemSaltoSTFT],
                                                                                                                                                                    dicionarioTiposJanelasSTFT[escolhaTipoJanelaSTFT],
                                                                                                                                                                    rmsQuadrosSerieTemporalAudioOriginal.min())
        
        frequenciasInstantaneasSerieTemporalAudioFinal, amplitudesFrequenciasInstantaneasSerieTemporalAudioFinal = obterFrequenciasAmplitudesSerieTemporalAudio(serieTemporalAudioFinal,
                                                                                                                                                                taxaAmostragemAudioFinal,
                                                                                                                                                                escolhaTamanhoJanelaSTFT,
                                                                                                                                                                dicionarioPorcentagensSaltosSTFT[escolhaPorcentagemSaltoSTFT],
                                                                                                                                                                dicionarioTiposJanelasSTFT[escolhaTipoJanelaSTFT],
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
        listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal = [int(round(frequencia, 0)) for frequencia in listaFrequenciasUteisSerieTemporalAudioOriginal]
        listaFrequenciasUteisArredondadasSerieTemporalAudioFinal = [int(round(frequencia, 0)) for frequencia in listaFrequenciasUteisSerieTemporalAudioFinal]

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
        if(max(listaDecibeisUteisArredondadosSerieTemporalAudioOriginal) > max(listaDecibeisUteisArredondadosSerieTemporalAudioFinal)):
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
                        markdown(f"**AJUSTE DE VOLUME:** Aumentar {str(menorDiferencaDecibeis)} decibéis")

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
                        markdown(f"**AJUSTE DE VOLUME:** Diminuir {str(menorDiferencaDecibeis)} decibéis")

        # Obtenção de frequências específicas selecionadas para exibição de respectivos decibéis
        with container(height=None, border=False):
            with spinner(text="Carregando campo..."):
                markdown("**ESCOLHA DE FREQUÊNCIAS PARA AJUSTE DE DECIBÉIS:**")
                obterFrequenciasEspecificasSelecionadasExibicaoDecibeis(listaFrequenciasUteisArredondadasSerieTemporalAudioOriginal, listaFrequenciasUteisArredondadasSerieTemporalAudioFinal,
                                                                        listaDecibeisUteisArredondadosSerieTemporalAudioOriginal, listaDecibeisUteisArredondadosSerieTemporalAudioFinal,
                                                                        listaFrequenciasComuns, menorDiferencaDecibeis)