import plotly.graph_objects as go


def visualize(data2D, data2DSubjects):
    """
    Params: 
        data2D: data frame with columns = ["TITLE", "COLOR", "X", "Y"]
        data2DSubjects: data frame with columns = ["TITLE", "COLOR", "SUBJECT", "X", "Y"]
    """

    paper_nodes = go.Scatter(
        x=data2D["X"],
        y=data2D["Y"],
        mode='markers',
        marker=dict(
            size=1,
            color=data2D["COLOR"]
        ),
        text=data2D["TITLE"],
        hoverinfo="text"
    )

    subject_nodes = go.Scatter(
        x=data2DSubjects["X"],
        y=data2DSubjects["Y"],
        opacity=0.75,
        text=data2DSubjects["SUBJECT"],
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=5,
            color=data2DSubjects["COLOR"]
        )
    )

    subject_texts = go.Scatter(
        x=data2DSubjects["X"],
        y=data2DSubjects["Y"],
        mode="text",
        text=data2DSubjects["TITLE"],
        textposition='top right',
        textfont=dict(size=10,color='black'),
    )

    data = [paper_nodes, subject_nodes, subject_texts]
    layout = go.Layout(
        height=1000,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend = False,
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()

def visualize3D(data2D, data2DSubjects):
    """
    Params: 
        data3D: data frame with columns = ["TITLE", "COLOR", "X", "Y", "Z"]
        data3DSubjects: data frame with columns = ["TITLE", "COLOR", "SUBJECT", "X", "Y", "Z"]
    """

    paper_nodes = go.Scatter3d(
        x=data2D["X"],
        y=data2D["Y"],
        z=data2D["Z"],
        mode='markers',
        marker=dict(
            size=1,
            color=data2D["COLOR"]
        ),
        text=data2D["TITLE"],
        hoverinfo="text"
    )

    subject_nodes = go.Scatter3d(
        x=data2DSubjects["X"],
        y=data2DSubjects["Y"],
        z=data2DSubjects["Z"],
        opacity=0.75,
        text=data2DSubjects["SUBJECT"],
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=5,
            color=data2DSubjects["COLOR"]
        )
    )

    subject_texts = go.Scatter3d(
        x=data2DSubjects["X"],
        y=data2DSubjects["Y"],
        z=data2DSubjects["Z"],
        mode="text",
        text=data2DSubjects["TITLE"],
        textposition='top right',
        textfont=dict(size=10,color='black'),
    )

    data = [paper_nodes, subject_nodes, subject_texts]
    layout = go.Layout(
        height=1000,
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend = False,
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()