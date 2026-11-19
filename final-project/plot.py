import os
import matplotlib.pyplot as plt
import pandas as pd

# Paleta de cores terrosas (verde e marrom)
COR_DESTAQUE = "#2D5A27"    # Verde floresta profundo (família líder da ordem)
COR_SECUNDARIA = "#5C8A58"  # Verde musgo (demais famílias)
COR_CAUDA = "#8C6239"       # Marrom terra (grupo "Outras")
COR_TEXTO = "#2B2118"       # Marrom café escuro
COR_FUNDO = "#FAF8F5"       # Off-white quente

# Diretório opcional para salvar os 28 gráficos sem travar a memória da IDE
PASTA_SAIDA = "graficos_por_ordem"

dicionario_ordens = {
    'Thysanoptera': ['Aeolothripidae', 'Thripidae', 'Phlaeothripidae'],
    'Coleoptera': [
        'Byturidae', 'Carabidae', 'Attelabidae', 'Buprestidae', 'Phalacridae',
        'Coccinellidae', 'Mycetophagidae', 'Ptinidae', 'Bostrichidae', 'Chrysomelidae',
        'Brentidae', 'Trogossitidae', 'Dermestidae', 'Laemophloeidae', 'Elateridae',
        'Staphylinidae', 'Tenebrionidae', 'Cerambycidae', 'Nitidulidae', 'Cantharidae',
        'Scarabaeidae', 'Oedemeridae', 'Cleridae', 'Silvanidae', 'Erotylidae',
        'Anthicidae', 'Meloidae', 'Cicindelidae', 'Passalidae', 'Curculionidae',
        'Anthribidae', 'Cryptophagidae', 'Latridiidae'
    ],
    'Diptera': [
        'Stratiomyidae', 'Drosophilidae', 'Calliphoridae', 'Piophilidae', 'Tachinidae',
        'Anthomyiidae', 'Syrphidae', 'Chloropidae', 'Tabanidae', 'Hippoboscidae',
        'Agromyzidae', 'Sciaridae', 'Fanniidae', 'Oestridae', 'Ulidiidae',
        'Asilidae', 'Culicidae', 'Ephydridae', 'Psychodidae', 'Tipulidae',
        'Dolichopodidae', 'Sarcophagidae', 'Bombyliidae', 'Tephritidae', 'Cecidomyiidae',
        'Simuliidae', 'Muscidae', 'Lonchaeidae', 'Polleniidae'
    ],
    'Ixodida': ['Ixodidae', 'Argasidae'],
    'Megaloptera': ['Corydalidae'],
    'Phasmida': ['Diapheromeridae', 'Pseudophasmatidae'],
    'Hymenoptera': [
        'Crabronidae', 'Mymaridae', 'Agaonidae', 'Formicidae', 'Siricidae',
        'Bembicidae', 'Dryinidae', 'Megachilidae', 'Andrenidae', 'Chrysididae',
        'Tenthredinidae', 'Cynipidae', 'Scelionidae', 'Cephidae', 'Ichneumonidae',
        'Pompilidae', 'Figitidae', 'Ampulicidae', 'Apidae', 'Eurytomidae',
        'Cimbicidae', 'Pamphiliidae', 'Vespidae', 'Diplolepididae', 'Anaxyelidae',
        'Evaniidae', 'Scoliidae', 'Diprionidae', 'Sphecidae', 'Braconidae', 'Halictidae'
    ],
    'Psocodea': [
        'Philopteridae', 'Menoponidae', 'Pthiridae', 'Linognathidae', 'Haematopinidae',
        'Trinotonidae', 'Hoplopleuridae', 'Lachesillidae', 'Trichodectidae',
        'Polyplacidae', 'Trogiidae', 'Liposcelididae'
    ],
    'Araneae': [
        'Desidae', 'Sparassidae', 'Araneidae', 'Gnaphosidae', 'Theridiidae',
        'Oxyopidae', 'Trechaleidae', 'Sicariidae'
    ],
    'Mesostigmata': ['Dermanyssidae', 'Macronyssidae'],
    'Symphyla_order_incertae_sedis': ['Scutigerellidae'],
    'Stylommatophora': ['Milacidae', 'Spiraxidae', 'Agriolimacidae', 'Helicidae', 'Limacidae'],
    'Julida': ['Julidae'],
    'Sarcoptiformes': [
        'Acaridae', 'Glycyphagidae', 'Psoroptidae', 'Sarcoptidae',
        'Pyroglyphidae', 'Carpoglyphidae'
    ],
    'Siphonaptera': ['Leptopsyllidae', 'Pulicidae', 'Ceratophyllidae', 'Tungidae'],
    'Isopoda': ['Chaetiliidae', 'Asellidae'],
    'Polydesmida': ['Paradoxosomatidae'],
    'Trombidiformes': [
        'Tetranychidae', 'Eriophyidae', 'Tarsonemidae', 'Demodicidae',
        'Tenuipalpidae', 'Penthaleidae'
    ],
    'Orthoptera': [
        'Trigonidiidae', 'Gryllotalpidae', 'Oecanthidae', 'Stenopelmatidae',
        'Pyrgomorphidae', 'Acrididae', 'Romaleidae', 'Gryllidae', 'Tettigoniidae'
    ],
    'Dermaptera': ['Chelisochidae', 'Forficulidae', 'Anisolabididae', 'Labiduridae'],
    'Neuroptera': ['Myrmeleontidae', 'Hemerobiidae', 'Chrysopidae'],
    'Odonata': ['Aeshnidae'],
    'Lepidoptera': [
        'Euteliidae', 'Cosmopterigidae', 'Nolidae', 'Erebidae', 'Lasiocampidae',
        'Yponomeutidae', 'Hesperiidae', 'Ypsolophidae', 'Saturniidae', 'Noctuidae',
        'Geometridae', 'Pieridae', 'Nepticulidae', 'Nymphalidae', 'Pyralidae',
        'Bombycidae', 'Cossidae', 'Galacticidae', 'Heliozelidae', 'Elachistidae',
        'Bucculatricidae', 'Depressariidae', 'Lycaenidae', 'Plutellidae',
        'Argyresthiidae', 'Oecophoridae', 'Bedelliidae', 'Glyphipterigidae',
        'Gracillariidae', 'Sphingidae', 'Prodoxidae', 'Riodinidae', 'Limacodidae',
        'Psychidae', 'Papilionidae', 'Choreutidae', 'Coleophoridae', 'Pterolonchidae',
        'Tortricidae', 'Dudgeoneidae', 'Crambidae', 'Tineidae', 'Sesiidae',
        'Gelechiidae', 'Incurvariidae', 'Notodontidae', 'Zygaenidae', 'Schreckensteiniidae'
    ],
    'Hemiptera': [
        'Tingidae', 'Aphrophoridae', 'Ischnorhinidae', 'Clastopteridae', 'Nabidae',
        'Lygaeidae', 'Membracidae', 'Geocoridae', 'Diaspididae', 'Anthocoridae',
        'Fulgoridae', 'Flatidae', 'Cicadidae', 'Gelastocoridae', 'Scutelleridae',
        'Reduviidae', 'Pentatomidae', 'Cimicidae', 'Triozidae', 'Coreidae',
        'Adelgidae', 'Miridae', 'Blissidae', 'Coccidae', 'Rhopalidae', 'Aphididae',
        'Thyreocoridae', 'Pyrrhocoridae', 'Pseudococcidae', 'Cydnidae', 'Cicadellidae',
        'Delphacidae', 'Aleyrodidae', 'Gerridae', 'Psyllidae', 'Monophlebidae'
    ],
    'Blattodea': [
        'Heterotermitidae', 'Archotermopsidae', 'Kalotermitidae', 'Pseudophyllodromiidae',
        'Blattellidae', 'Blaberidae', 'Ectobiidae', 'Corydiidae', 'Rhinotermitidae',
        'Termitidae', 'Blattidae'
    ],
    'Embioptera': ['Oligotomidae'],
    'Mantodea': ['Mantidae'],
    'Symphypleona': ['Bourletiellidae']
}

def plotar_ordem(
        df_ordem: pd.DataFrame,
        nome_ordem: str,
        max_barras: int = 12,
        salvar: bool = True,
        exibir: bool = False,
    ):
        """Gera o gráfico de barras horizontais ordenado para uma única ordem taxonômica."""
        df_ord = df_ordem.sort_values("quantidade_imagens", ascending=False).reset_index(drop=True)
        total_ordem = df_ord["quantidade_imagens"].sum()
        n_familias_total = len(df_ord)

        if total_ordem == 0 or n_familias_total == 0:
            return

        df_ord["percentual"] = (df_ord["quantidade_imagens"] / total_ordem) * 100
        lider_nome = df_ord.loc[0, "categoria"]
        lider_pct = df_ord.loc[0, "percentual"]

        # Agrupa a cauda longa apenas se a ordem tiver mais de `max_barras` famílias
        if n_familias_total > max_barras:
            principais = df_ord.iloc[: max_barras - 1].copy()
            menores = df_ord.iloc[max_barras - 1 :]
            outros = pd.DataFrame({
                "categoria": [f"Outras ({len(menores)} famílias)"],
                "quantidade_imagens": [menores["quantidade_imagens"].sum()],
                "percentual": [menores["percentual"].sum()],
            })
            df_plot = pd.concat([principais, outros], ignore_index=True)
        else:
            df_plot = df_ord.copy()

        # Inverte para manter a maior barra no topo
        df_plot = df_plot.iloc[::-1].reset_index(drop=True)

        cores = [
            COR_CAUDA if cat.startswith("Outras (")
            else (COR_DESTAQUE if cat == lider_nome else COR_SECUNDARIA)
            for cat in df_plot["categoria"]
        ]

        # Altura proporcional ao número de barras para evitar achatamento ou excesso de espaço
        n_barras = len(df_plot)
        altura_fig = max(2.6, 1.35 + n_barras * 0.44)
        fig, ax = plt.subplots(figsize=(10, altura_fig), dpi=150)
        fig.patch.set_facecolor(COR_FUNDO)
        ax.set_facecolor(COR_FUNDO)

        barras = ax.barh(
            df_plot["categoria"],
            df_plot["quantidade_imagens"],
            color=cores,
            height=0.62 if n_barras > 2 else 0.45,
            edgecolor="none",
        )

        # Remoção de ruídos visuais (Princípios de Tufte)
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#C4B7A6")
        ax.spines["left"].set_linewidth(0.8)

        ax.xaxis.set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=COR_TEXTO, pad=8)

        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == lider_nome:
                tick_label.set_fontweight("bold")
                tick_label.set_color(COR_DESTAQUE)

        # Rótulos diretos de dados
        max_val = df_plot["quantidade_imagens"].max()
        for barra, (_, row) in zip(barras, df_plot.iterrows()):
            largura = barra.get_width()
            texto = f" {row['quantidade_imagens']:,.0f} ({row['percentual']:.1f}%)".replace(",", ".")
            is_top = row["categoria"] == lider_nome

            ax.text(
                largura + (max_val * 0.012),
                barra.get_y() + barra.get_height() / 2,
                texto,
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold" if is_top else "normal",
                color=COR_DESTAQUE if is_top else COR_TEXTO,
            )

        ax.set_xlim(0, max_val * 1.26)

        # Título dinâmico orientado a insights
        if n_familias_total == 1:
            titulo = f"{nome_ordem}: {lider_nome} representa 100% do acervo da ordem"
        else:
            titulo = f"{nome_ordem}: {lider_nome} lidera com {lider_pct:.1f}% das imagens da ordem"

        sufixo_agrupamento = (
            f" | top {max_barras - 1} + cauda agrupada" if n_familias_total > max_barras else ""
        )
        subtitulo = (
            f"Total de {total_ordem:,.0f} imagens distribuídas em {n_familias_total} "
            f"{'família' if n_familias_total == 1 else 'famílias'}{sufixo_agrupamento}"
        ).replace(",", ".")

        ax.set_title(
            f"{titulo}\n",
            loc="left",
            fontsize=12.5,
            fontweight="bold",
            color=COR_TEXTO,
            pad=14,
        )
        # Adiciona o subtítulo logo abaixo do título principal sem depender de coordenadas fixas da figura
        ax.text(
            0,
            1.03,
            subtitulo,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#5E503F",
            ha="left",
            va="bottom",
        )

        plt.tight_layout()

        if salvar:
            caminho = os.path.join(PASTA_SAIDA, f"{nome_ordem}.png")
            plt.savefig(caminho, dpi=150, bbox_inches="tight", facecolor=COR_FUNDO)

        if exibir:
            plt.show()
        else:
            plt.close(fig)

def plota_por_ordem(df_prep):
    familia_para_ordem = {
        familia: ordem
        for ordem, familias in dicionario_ordens.items()
        for familia in familias
    }

    df_prep["quantidade_imagens"] = pd.to_numeric(df_prep["quantidade_imagens"])
    df_prep["ordem"] = df_prep["categoria"].map(familia_para_ordem).fillna("Não mapeada")

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # 3. Execução em lote respeitando a ordem definida no dicionário
    for ordem_nome in dicionario_ordens.keys():
        subset = df_prep[df_prep["ordem"] == ordem_nome]
        if not subset.empty:
            # Altere exibir=True se estiver em Jupyter Notebook e quiser renderizar todos na tela
            plotar_ordem(subset, ordem_nome, max_barras=12, salvar=True, exibir=False)

    print(f"Gráficos gerados com sucesso na pasta '{PASTA_SAIDA}/'.")