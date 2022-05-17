import json
from xml.etree.ElementInclude import include
import requests

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

import utils.constants as constants
import preparation.extracting as extracting

HEADERS = {}

def build_semantic_scholar_reference_tree(paper_id, papers, depth=3, include_leaf_connections=False):
    """
    Scrapes a tree of references based on a source paper and a given depth. Saves the scraped references in a dictionary.

    Params: 
        paper_id: id specifying the root paper
        papers: dictionary with paper_ids as keys and paper_details as values
        depth: depth of references to be considered
    """

    if depth >= 0:
        if depth == 0 and not include_leaf_connections:
            return

        paper_references_url = "https://api.semanticscholar.org/graph/v1/paper/%s/references?fields=title,abstract,authors,year,referenceCount,citationCount,fieldsOfStudy"
        paper_references_response = requests.get(paper_references_url % (paper_id), headers=HEADERS)
        paper_references_list_raw = json.loads(paper_references_response.content.decode('utf-8'))['data']

        references_id_list = []
        
        for raw_paper in paper_references_list_raw:
            child_paper = raw_paper['citedPaper']
            child_paper_id = child_paper['paperId']

            if child_paper_id is not None:
                if depth == 0 and child_paper_id in papers:
                    references_id_list.append(child_paper_id)
                    papers[paper_id]['referencesIds'] = references_id_list
                    continue

                references_id_list.append(child_paper_id)

                if child_paper_id not in papers:

                    current_paper = {
                        'title': child_paper['title'], 
                        'abstract': child_paper['abstract'], 
                        'year': child_paper['year'], 
                        'authors': child_paper['authors'], 
                        'referenceCount': child_paper['referenceCount'], 
                        'citationCount': child_paper['citationCount'],
                        'fieldsOfStudy': extracting.clean_s2orc_categories(child_paper['fieldsOfStudy']),
                        'referencesIds': []
                    }

                    papers[child_paper_id] = current_paper
                    build_semantic_scholar_reference_tree(child_paper_id, papers, depth-1, include_leaf_connections)

        papers[paper_id]['referencesIds'] = references_id_list

def generate_adjacency_matrix_from_references(all_references):
    """
    Creates an adjacency matrix from a dictionary of papers. This matrix can later be used to visualize a reference tree.

    Params:
        all_references: dictionary of papers that are in a reference relation (key: paper_id, value: dict of paper details)
    
    Returns:
        adjacency_matrix: binary matrix, where 1 represents a reference relation between two papers
    """

    paper_ids = all_references.keys()
    num_entries = len(all_references)
    paper_id2index = dict(zip(paper_ids, range(num_entries)))
    adjacency_matrix = np.zeros(shape = (num_entries, num_entries))

    for paper_id, paper in all_references.items():
        index_i = paper_id2index[paper_id]

        for reference_id in paper["referencesIds"]:
            if reference_id in paper_id2index:
                index_j = paper_id2index[reference_id]
                adjacency_matrix[index_i][index_j] = 1

    return adjacency_matrix

def visualize_reference_tree(all_references):
    """
    Draws a reference tree from a dictionary of papers.

    Params:
        all_references: dictionary of papers that are in a reference relation (key: paper_id, value: dict of paper details)
    """
    paper_id_list = list(all_references.keys())
    num_entries = len(paper_id_list)

    adjacency_matrix = generate_adjacency_matrix_from_references(all_references)

    graph = nx.Graph(adjacency_matrix)
    position = nx.spring_layout(graph)
    fig = plt.figure(1, figsize=(200, 200), dpi = 50)
    nx.draw(graph, node_size = 400, font_size=30, pos = position, with_labels = True, labels = dict(zip(range(num_entries), paper_id_list)))

def create_hovertext(paper_id, paper): 

    title = paper["title"]
    authors = ", ".join([author["name"] for author in paper["authors"]])
    citations = str(paper["citationCount"])
    subjects = str(paper["fieldsOfStudy"])

    return "%s<br>Authors: %s<br>ID: %s<br>Citations: %s<br>Subjects: %s" % (title, authors, paper_id, citations, subjects)

def visualize_reference_tree_plotly(all_references):
    """
    Draws a reference tree from a dictionary of papers.
    Plotly is used to make the visualization interactive

    Params:
        all_references: dictionary of papers that are in a reference relation (key: paper_id, value: dict of paper details)
    """

    paper_id_list = list(all_references.keys())
    citation_counts = [paper["citationCount"] if paper["citationCount"] is not None else 0 for paper in all_references.values()]
    titles = [paper["title"] for paper in all_references.values()]
    relative_citation_counts = ((MinMaxScaler().fit_transform(np.array(citation_counts).reshape((-1, 1))) + 1) * 10).ravel().tolist()
    subjects = [paper["fieldsOfStudy"][0] if paper["fieldsOfStudy"] is not None else "None" for paper in all_references.values()]
    subject_colors = [constants.SUBJECTS2COLOR[subject] for subject in subjects]

    hovertext = [ create_hovertext(paper_id, paper) for paper_id, paper in all_references.items() ]

    num_entries = len(paper_id_list)

    adjacency_matrix = generate_adjacency_matrix_from_references(all_references)
    graph = nx.Graph(adjacency_matrix)
    positions = nx.spring_layout(graph, seed=1234)

    x_nodes = [positions[i][0] for i in range(num_entries)]
    y_nodes = [positions[i][1] for i in range(num_entries)]

    edge_list = graph.edges()

    x_edges = []
    y_edges = []

    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [positions[edge[0]][0],positions[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [positions[edge[0]][1],positions[edge[1]][1],None]
        y_edges += y_coords

    trace_edges = go.Scatter(
        x=x_edges,
        y=y_edges,
        mode='lines',
        line=dict(
            color='gray',
            width=1
        ),
        hoverinfo='none',
        showlegend=False
    )

    trace_nodes = go.Scatter(
        x=x_nodes,
        y=y_nodes,
        mode='markers',
        marker=dict(
            size=relative_citation_counts,
            color=subject_colors
        ),
        text=titles,
        showlegend=False,
        hoverinfo="text",
        hovertext=hovertext,
        customdata=list(zip(all_references.keys(), all_references.values()))
    )

    data = [trace_edges, trace_nodes]

    for subject in set(subjects):
        data.append(
            go.Scatter(
                x=[None], 
                y=[None], 
                mode='markers',
                marker=dict(size=10, color=constants.SUBJECTS2COLOR[subject]),
                legendgroup='Subjects',
                showlegend=True,
                name=subject
            )
        )

    layout = go.Layout(
        height=800,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def visualize_reference_tree_plotly3d(all_references):
    """
    Draws a reference tree from a dictionary of papers.
    Plotly is used to make the visualization interactive

    Params:
        all_references: dictionary of papers that are in a reference relation (key: paper_id, value: dict of paper details)
    """

    paper_id_list = list(all_references.keys())
    citation_counts = [paper["citationCount"] for paper in all_references.values()]
    titles = [paper["title"] for paper in all_references.values()]
    relative_citation_counts = ((MinMaxScaler().fit_transform(np.array(citation_counts).reshape((-1, 1))) + 1) * 10).ravel().tolist()
    subjects = [paper["fieldsOfStudy"][0] if paper["fieldsOfStudy"] is not None else "None" for paper in all_references.values()]
    subject_colors = [constants.SUBJECTS2COLOR[subject] for subject in subjects]

    hovertext = [paper["title"] + "<br>Citations: " + str(paper["citationCount"]) + "<br>Subjects: " +  str(paper["fieldsOfStudy"]) for paper in all_references.values()]

    num_entries = len(paper_id_list)

    adjacency_matrix = generate_adjacency_matrix_from_references(all_references)
    graph = nx.Graph(adjacency_matrix)
    positions = nx.spring_layout(graph, dim = 3)

    x_nodes = [positions[i][0] for i in range(num_entries)]
    y_nodes = [positions[i][1] for i in range(num_entries)]
    z_nodes = [positions[i][2] for i in range(num_entries)]

    edge_list = graph.edges()

    x_edges = []
    y_edges = []
    z_edges = []

    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [positions[edge[0]][0],positions[edge[1]][0],None]
        x_edges += x_coords

        y_coords = [positions[edge[0]][1],positions[edge[1]][1],None]
        y_edges += y_coords

        z_coords = [positions[edge[0]][2],positions[edge[1]][2],None]
        z_edges += z_coords

    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(
            color='gray', 
            width=1
        ),
        hoverinfo='none'
    )

    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        marker=dict(
            size=relative_citation_counts,
            color=subject_colors
        ),
        text=titles,
        hoverinfo="text",
        hovertext=hovertext
    )

    data = [trace_edges, trace_nodes]
    layout = go.Layout(
        height=800,
        hovermode='closest', 
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', 
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )

    fig = go.Figure(data=data, layout=layout)
    return fig