import json  # for saving and parsing json files
import pandas  # for reading and dumping data
import os


def scrape_faculty_data():
    import scholarly

    # List of faculty names and Google Scholar IDs
    faculty_in_department: list[dict] = pandas.read_csv("faculty.csv").to_dict(
        "records"
    )

    # Make data directory if it doesn't already exist
    os.makedirs("data", exist_ok=True)

    # Get publications for every faculty member
    for faculty in faculty_in_department:
        if not pandas.isnull(faculty["id"]):  # make sure there is an ID in the dict
            # Get and fill the info
            author = scholarly.scholarly.fill(
                scholarly.scholarly.search_author_id(faculty["id"])
            )
            # Extract publications
            faculty_pubs = [
                publication["bib"] for publication in author["publications"]
            ]
            # Save to JSON
            with open("data/" + faculty["name"] + ".json", "w", encoding="utf-8") as f:
                json.dump(
                    sorted(faculty_pubs, key=lambda x: x["title"]),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )


def visualize_faculty_data():
    import matplotlib.colors  # for getting pretty colors
    import matplotlib.pyplot  # for converting rgb to hex
    import numpy  # for generic operations
    import plotly.express  # plotly
    import sentence_transformers  # embeddings
    import sklearn.decomposition  # orient tsne
    import sklearn.manifold  # make a tsne
    from numpy.typing import NDArray  # for type hinting
    from sentence_transformers import SentenceTransformer  # for type hinting
    from plotly.graph_objs import Figure  # for type hinting
    from pandas import DataFrame  # for type hinting

    # Identify all the json files
    path_to_json: str = "./data/"
    json_files: list[str] = [
        pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith(".json")
    ]

    # Dump all the json files into a single dataframe
    all_the_data: DataFrame = pandas.DataFrame()
    for json_file in sorted(json_files, key=str.casefold):
        with open(os.path.join(path_to_json, json_file)) as json_file_path:
            json_dict: dict = json.load(json_file_path)
            json_as_df: DataFrame = pandas.DataFrame.from_dict(json_dict)
            json_as_df["faculty"] = json_file.replace(".json", "")
            all_the_data: DataFrame = pandas.concat([all_the_data, json_as_df], axis=0)

    # Re-index the dataframe because it appears to eb necessary
    all_the_data.reset_index(inplace=True)

    # Embed titles from publications
    model: SentenceTransformer = sentence_transformers.SentenceTransformer(
        "all-mpnet-base-v2"
    )
    embeddings: NDArray[float] = model.encode(
        all_the_data["title"].to_list(), show_progress_bar=True
    )

    # Boil down teh data into a 2D plot
    tsne_embeddings: NDArray[float] = sklearn.manifold.TSNE(
        n_components=2, random_state=42, verbose=0
    ).fit_transform(embeddings)
    pca_embeddings: NDArray[float] = sklearn.decomposition.PCA(
        n_components=2, random_state=42
    ).fit_transform(tsne_embeddings)
    all_the_data["x"] = pca_embeddings[:, 0]
    all_the_data["y"] = pca_embeddings[:, 1]

    # Make some pretty colors!
    colors: list[str] = [
        matplotlib.colors.to_hex(x)
        for x in matplotlib.pyplot.cm.gist_rainbow(
            numpy.linspace(0, 1, len(json_files))
        )
    ]

    # Plot the embeddings
    fig: Figure = plotly.express.scatter(
        all_the_data,
        x="x",
        y="y",
        hover_data=["title"],
        color="faculty",
        symbol="faculty",
        color_discrete_sequence=colors,
    )

    # Make sure the axes are appropriately scaled
    fig.update_xaxes(
        visible=False,
        autorange=False,
        range=[
            numpy.min(pca_embeddings[:, 0]) * 1.05,
            numpy.max(pca_embeddings[:, 0]) * 1.05,
        ],
    )

    fig.update_yaxes(
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        range=[
            numpy.min(pca_embeddings[:, 1]) * 1.05,
            numpy.max(pca_embeddings[:, 1]) * 1.05,
        ],
    )

    # Reset the layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(y=0.5),
        plot_bgcolor="#191C1F",
    )

    # Remove the logo
    fig.show(config=dict(displaylogo=False))

    # Save the file
    fig.write_html("index.html")
