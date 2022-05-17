const allLabels = [
    "Agricultural Science", "Arts and Humanities", "Biochemistry", "Biology", "Business",  "Chemical Engineering", "Chemistry", 
    "Computer Science", "Decision Sciences", "Dentistry", "Earth and Planetary Sciences", "Economics", "Energy Sciences", "Engineering", 
    "Environmental Science", "Geology","Geography", "Health Professions","History","Immunology", "Material Sciences", "Mathematics", 
    "Medical Science", "MULT", "Neuroscience", "Nursing","Political Science","Pharmacy","Philosophy","Physics", "Psychology", "Social Sciences", 
    "Veterinary Science", "Electrical Engineering", "Statistics", "Quantitative Biology", "Quantitative Finance", "Civil Engineering"
];

const getPaperResults = async (searchQuery) => {
    return await eel.get_paper_results(searchQuery)();
}

const getPaperDetails = async (paperId) => {
    return await eel.get_paper_details(paperId)();
}

const getSimilarPapersSubjects = async (paper, subjects, keywords, top_n) => {
    return await eel.get_similar_papers_subjects(paper, subjects, keywords, top_n)();
}

const getSimilarPapersNeighborhood = async (paperId, paperDetails, keywords, top_n) => {
    return await eel.get_similar_papers_neighborhood(paperId, paperDetails, keywords, top_n)();
}

const getSimilarPapersAll = async (paper, subjects, keywords, top_n) => {
    return await eel.get_similar_papers_all(paper, subjects, keywords, top_n)();
}

const getReferenceTree = async (paperId, paperDetails) => {
    return await eel.get_reference_tree(paperId, paperDetails)();
}

const getPdfInformation = async () => {
    return await eel.get_pdf_information()();
}

const getPaperSubject = async (abstract) => {
    return await eel.get_paper_subject(abstract)();
}

async function displaySimilarPapers(similarPapers, by_subjects = false) {

    let recommendationsGridCells = document.createElement("div");

    if (by_subjects) {
        for (const [subject, value] of Object.entries(similarPapers)) {

            let headline = document.createElement("h3"); 
            headline.textContent = subject;
            recommendationsGridCells.append(headline, document.createElement("br")); 
    
            for (let entry of value) {
                id = entry[0]
                title = entry[3]
                distance = entry[1]
                abstract = entry[4]
    
                recommendationsGridCells.appendChild(generateRecommendationsGridCell(id, title, abstract, subject, distance));
            }
    
            recommendationsGridCells.appendChild(document.createElement("br"));
        }
    } else {
        for (const entry of similarPapers) {
            id = entry[0]
            title = entry[3]
            distance = entry[1]
            abstract = entry[4]
            subject = entry[2]
    
            recommendationsGridCells.appendChild(generateRecommendationsGridCell(id, title, abstract, subject, distance));
        }
    }

    let contentHTML = document.getElementById('content');
    contentHTML.innerHTML = ``;
    contentHTML.appendChild(recommendationsGridCells);
    contentHTML.scrollIntoView({behavior: "smooth"});
}

async function displaySimilarPapersSubjects(abstract, subjects, keywords = null, top_n = 10) {
    displayLoadingAnimation("content", false);
    similarPapers = await getSimilarPapersSubjects({"abstract": abstract}, subjects, keywords, top_n);
    displaySimilarPapers(similarPapers, true);
}

async function displaySimilarPapersNeighborhood(paperId, paperDetails, keywords = null, top_n = 100) {
    displayLoadingAnimation("content", false);
    similarPapers = await getSimilarPapersNeighborhood(paperId, paperDetails, keywords, top_n);
    displaySimilarPapers(similarPapers);
}

async function displaySimilarPapersAll(abstract, subjects) {
    displayLoadingAnimation("content", false);
    similarPapers = await getSimilarPapersAll({"abstract": abstract}, subjects);
    displaySimilarPapers(similarPapers);
}

function displayPaperDetails(paperId, paperDetails) {

    let background = document.getElementById('background');
    let authorsString = "Authors unknown";
    let subjectString = "Subject unknown";

    if (paperDetails.authorsString != null) {
        authorsString = paperDetails.authorsString;
    } else if(paperDetails.authors != null) {
        authorsString = paperDetails.authors.map(author => author.name).join(", ");
    }

    if (paperDetails.fieldsOfStudy != null && paperDetails.fieldsOfStudy.length > 0) {
        subjectString = paperDetails.fieldsEstimated ? "Estimated: " : ""
        subjectString = subjectString + paperDetails.fieldsOfStudy.join(", ");
    }

    background.innerHTML = 
        `<div class="details-background">

            <h3>${paperDetails.title}</h3>
            <br>

            <p>${authorsString}</p>
            <br>

            <div class="details">

                <h4>Abstract</h4>
                <br>
                <p>${paperDetails.abstract}</p>
                <br>

                <h4>Subjects</h4>
                <br>
                <p>${subjectString}</p>
                <br>
            </div>
            <h4>Tools</h4>
            <br>
            <div class="btn-wrapper" id="btn-wrapper">
                <div class="btn" id="similar-papers-btn">Similar Papers</div>
                <div class="btn" id="similar-papers-advanced-btn">Similar Papers (Advanced)</div>
            </div>
        </div>
        <div id="content"></div>
        `;

    if (paperId != null) {
        let referenceTreeBtn = document.createElement("div");
        referenceTreeBtn.className = "btn";
        referenceTreeBtn.id = "reference-tree-btn";
        referenceTreeBtn.innerHTML = "Reference Tree";
        referenceTreeBtn.onclick = () => { displayReferenceTree(paperId, paperDetails)};

        let neighborhoodBtn = document.createElement("div");
        neighborhoodBtn.className = "btn";
        neighborhoodBtn.id = "neighborhood-btn";
        neighborhoodBtn.innerHTML = "Neighborhood";
        neighborhoodBtn.onclick = () => { displayOptionsForSimilarPapersNeighborhood(paperId, paperDetails)};

        let btnWrapper = document.getElementById("btn-wrapper");
        btnWrapper.append(neighborhoodBtn, referenceTreeBtn);
    }

    let similarPapersBtn = document.getElementById("similar-papers-btn"); 
    let similarPapersAdvancedBtn = document.getElementById("similar-papers-advanced-btn"); 

    similarPapersBtn.onclick = () => { displayOptionsForSimilarPapers(paperDetails.abstract)};
    similarPapersAdvancedBtn.onclick = () => {displayAdvancedOptionsForSimilarPapers(paperDetails.abstract)};
}

const displaySearchResults = async (searchQuery) => {

    papers = await getPaperResults(searchQuery);

    let background = document.getElementById('background');
    background.innerHTML = "";

    let searchBackground = document.createElement("div");
    searchBackground.className = "search-background";
    searchBackground.innerHTML = `<h1>Results for "${searchQuery}"</h1>`;

    let searchGrid = document.createElement("div");
    searchGrid.className = "search-grid";

    for (const [key, value] of Object.entries(papers)) {
        if(value.title != null && value.abstract != null){
            searchGrid.appendChild(generateSearchGridCell(key, value));
        }
    }

    searchBackground.appendChild(searchGrid);
    background.appendChild(searchBackground);
}

function displayPaperParsingDialog(title, authors, abstract) {

    let background = document.getElementById('background');

    authorsString = "No authors found";
    if (authors)
        authorsString = authors.map(author => author.name).join(",  ");
        
    background.innerHTML = 
        `<div class="parsing-wrapper">
            <h3>Automatically Parsed Information</h3>
            <br>

            <h4>Title</h4>
            <br>
            <textarea id="title-input" >${title}</textarea>
            <br>
            <br>

            <h4>Authors</h4>
            <br>
            <textarea id="authors-input" >${authorsString}</textarea>
            <br>
            <br>
            
            <h4>Abstract</h4>
            <br>
            <textarea id="abstract-input" >${abstract}</textarea>
            <br>
            <br>

            <div class="btn" id="custom-paper-btn">Confirm Inputs</div>
        </div>`;

    let customPaperBtn = document.getElementById("custom-paper-btn");
    customPaperBtn.onclick = async () => {

        let title = document.getElementById("title-input").value;
        let authors = document.getElementById("authors-input").value;
        let abstract = document.getElementById("abstract-input").value;
        let subjects = await getPaperSubject(abstract);

        displayPaperDetails(null, {title: title, authorsString: authors, abstract: abstract, fieldsOfStudy: [subjects], fieldsEstimated: true});
    }
}

const displayHome = () => {
    background = document.getElementById('background');

    background.innerHTML = 
        `<div class="home-wrapper">
            <div class="search-wrapper">
                <i class="fa fa-search" id="search-icon"></i>
                <input 
                    type="text" 
                    id="search-input" 
                    name="search-input" 
                    data-provide="typeahead" 
                    data-items="4" 
                    placeholder="Type keywords and hit enter" 
                    onkeypress="onKeyListener(event)"
                >
            </div>
            <div class="upload-wrapper">
                <input type="file" id="upload" name="upload" accept=".pdf">
                <label id="upload-btn" for="upload"><div class="upload-inner">Upload a Paper (.pdf)</div></label>
            </div>
        </div>`;
}

const displayLoadingAnimation = (elementId, fixed = true) => {
    background = document.getElementById(elementId);
    background.innerHTML = fixed ? 
        `<div class="loading-background fixed"><div class="loading-animation"></div></div>`:
        `<div class="loading-background"><div class="loading-animation"></div></div>`;
}

async function processFileInput(){
    displayLoadingAnimation("background");
    information = await getPdfInformation();
    displayPaperParsingDialog(information.title, information.authors, information.abstract, information.references);
}

const generateRecommendationsGridCell = (id, title, abstract, subject, distance) => {

    let recommendationsGridCell = document.createElement("div"); 
    recommendationsGridCell.className = "recommendations-grid-cell";

    let gridInfo = document.createElement("div"); 
    gridInfo.className = "grid-info"; 
    gridInfo.innerHTML = `<p>${distance.toFixed(2)} | ${subject}</p>`;

    let gridTitle = document.createElement("h4"); 
    gridTitle.className = "clickable";
    gridTitle.textContent = title;

    gridTitle.onclick = async () => {

         // SemanticScholar ids have a length of 40
         // In this case we complete the data by making an API call
        if (id.length == 40) {
            displayLoadingAnimation("content", false);
            let paperDetails = await getPaperDetails(id);

            if (paperDetails.fieldsOfStudy == null) {
                paperDetails.fieldsOfStudy = [await getPaperSubject(paperDetails.abstract)];
                paperDetails.fieldsEstimated = true;
            }

            displayPaperDetails(id, paperDetails);
        } else {
            displayPaperDetails(null, {title: title, authorsString: null, abstract: abstract, fieldsOfStudy: [subject]});
        }
    }

    let gridAbstract = document.createElement("div"); 
    gridAbstract.className = "grid-abstract"; 
    gridAbstract.innerHTML = `<p>${abstract}</p>`; 

    recommendationsGridCell.append(
        gridInfo, gridTitle, 
        document.createElement("br"), 
        gridAbstract, 
        document.createElement("br")
    );

    return recommendationsGridCell; 
}

const generateLabelButtons = (labels) => {
    return labels.map((label, index) => 
        `<div class="btn-subject">
            <input type="checkbox" class="label-checkbox" id="label-${index}" name="label-btn-${index}">
            <label for="label-${index}" id="label-btn-label-${index}">${label}</label>
        </div>`
        ).join("");
}

function extractSubjectLabels() {
    let subjects = []
    let checkboxes = document.getElementsByClassName('label-checkbox');

    for(let i = 0; i < checkboxes.length; i++){
        if(checkboxes[i].checked){
            checkboxLabel = document.getElementById('label-btn-label-' + i);
            subjects.push(checkboxLabel.innerHTML);
        }
    }

    return subjects;
}

const displayOptionsForSimilarPapers = (abstract) => {

    let labelButtons = generateLabelButtons(allLabels);
    let content = document.getElementById("content");

    content.style.display = "block";
    content.innerHTML = `
        <h3>Select Subjects of Interest</h3>
        <br>
        <div class="btn-wrapper-subjects">${labelButtons}</div>
        <br>
        <br>
        <h3>Select Ranking Approach</h3>
        <br>
        <div class="btn-wrapper">
            <div class="btn" id="rank-subjects-btn">Rank per subject</div>
            <div class="btn" id="rank-all-btn">Rank all</div>
        </div>
    `;

    let rankSubjectBtn = document.getElementById("rank-subjects-btn");
    let rankAllBtn = document.getElementById("rank-all-btn");

    rankSubjectBtn.onclick = () => {
        let subjects = extractSubjectLabels();
        displaySimilarPapersSubjects(abstract, subjects);
    }

    rankAllBtn.onclick = () => {
        let subjects = extractSubjectLabels();
        displaySimilarPapersAll(abstract, subjects);
    }

    content.scrollIntoView({behavior: "smooth"});
}

const displayAdvancedOptionsForSimilarPapers = (abstract) => {

    let labelButtons = generateLabelButtons(allLabels);
    let content = document.getElementById("content");

    content.style.display = "block";
    content.innerHTML = `
        <h3>Select Subjects of Interest</h3>
        <br>
        <div class="btn-wrapper-subjects">${labelButtons}</div>
        <br>
        <h3>Enter Keywords of Interest</h3>
        <br>
        <textarea id="keywords-input" placeholder="Enter comma-separated keywords..."></textarea>
        <br>
        <br>
        <h3>Number of Results</h3>
        <br>
        <textarea id="number-results-input" placeholder="Enter number of results..."></textarea>
        <br>
        <br>
        <h3>Select Ranking Approach</h3>
        <br>
        <div class="btn-wrapper">
            <div class="btn" id="rank-subjects-btn">Rank per subject</div>
            <div class="btn" id="rank-all-btn">Rank all</div>
        </div>
    `;

    let rankSubjectBtn = document.getElementById("rank-subjects-btn");
    rankSubjectBtn.onclick = () => {
        
        let keywords = document.getElementById("keywords-input").value;
        keywords = keywords.length < 1 ? null : keywords.split(",").map(keyword => keyword.trim());

        let top_n = document.getElementById("number-results-input").value;
        top_n = parseInt(top_n) || 10;

        let subjects = extractSubjectLabels();
        displaySimilarPapersSubjects(abstract, subjects, keywords, top_n);
    }

    let rankAllBtn = document.getElementById("rank-all-btn");
    rankAllBtn.onclick = () => {
        
        let keywords = document.getElementById("keywords-input").value;
        keywords = keywords.length < 1 ? null : keywords.split(",").map(keyword => keyword.trim());

        let top_n = document.getElementById("number-results-input").value;
        top_n = parseInt(top_n) || 10;

        let subjects = extractSubjectLabels();
        displaySimilarPapersAll(abstract, subjects, keywords, top_n);
    }

    content.scrollIntoView({behavior: "smooth"});
}

const displayOptionsForSimilarPapersNeighborhood = (paperId, paperDetails) => {

    let content = document.getElementById("content");

    content.style.display = "block";
    content.innerHTML = `
        <h3>Enter Keywords of Interest</h3>
        <br>
        <textarea id="keywords-input" placeholder="Enter comma-separated keywords..."></textarea>
        <br>
        <br>
        <h3>Number of Results</h3>
        <br>
        <textarea id="number-results-input" placeholder="Enter number of results..."></textarea>
        <br>
        <br>
        <div class="btn-wrapper">
            <div class="btn" id="search-btn">Start Search</div>
        </div>
    `;

    let startSearchBtn = document.getElementById("search-btn");
    startSearchBtn.onclick = () => { 
        
        let keywords = document.getElementById("keywords-input").value;
        keywords = keywords.length < 1 ? null : keywords.split(",").map(keyword => keyword.trim());

        let top_n = document.getElementById("number-results-input").value;
        top_n = parseInt(top_n) || 100;
        
        displaySimilarPapersNeighborhood(paperId, paperDetails, keywords, top_n) 
    }

    content.scrollIntoView({behavior: "smooth"});
}

const generateSearchGridCell = (paperId, paperDetails) => {

    let title = paperDetails.title;
    let abstract = paperDetails.abstract;
    let authors = paperDetails.authors;
    let authorsString = "Authors unknown";
    
    if(title != null && title.length > 100){ title = title.substring(0, 100) + "..."; }

    if(authors != null)
        authorsString = authors.length > 3 ? authors.slice(0,3).map(author => author.name).join(",  ").concat(" et al.") : authors.map(author => author.name).join(", ");

    let searchGridCell = document.createElement("div");
    searchGridCell.className = "search-grid-cell";

    let searchGridAbstractTitle = document.createElement("h4");
    let searchGridAbstractParagraph = document.createElement("p");
    searchGridAbstractTitle.textContent = title;
    searchGridAbstractParagraph.textContent = abstract;
    
    let searchGridAbstract = document.createElement("div");
    searchGridAbstract.className = "search-grid-abstract";
    searchGridAbstract.append(searchGridAbstractTitle, searchGridAbstractParagraph);
    
    let searchGridSummary = document.createElement("div");
    searchGridSummary.innerHTML = `
            <p class="grid-authors">${authorsString}</p>
            <p class="grid-citations">${paperDetails.citationCount} citations</p>
        `;
    searchGridSummary.className = "search-grid-summary";

    searchGridCell.append(searchGridAbstract, searchGridSummary);

    searchGridCell.onclick = () => {displayPaperDetails(paperId, paperDetails)}
    return searchGridCell;
}

async function displayReferenceTree(paperId, paperDetails) {

    let content = document.getElementById("content");
    content.style.display = "block";

    displayLoadingAnimation("content", false);
    content.scrollIntoView({behavior: "smooth"});

    referenceTreeJSON = await getReferenceTree(paperId, paperDetails);
    content.innerHTML = `
            <h3>Reference Tree</h3>
            <br>
            <div class="reference_tree_plot_container">
                <div id="reference-tree-plot"></div>
            </div>
        `;

    const data = referenceTreeJSON.data
    const layout = referenceTreeJSON.layout
    const config = {responsive: true}

    Plotly.newPlot("reference-tree-plot", data, layout, config);
    content.scrollIntoView({behavior: "smooth"});

    document.getElementById("reference-tree-plot").on('plotly_click', function(data){

        let paperData = data.points[0].customdata;

        if (paperData != null) {
            let paperId = paperData[0];
            let paperDetails = paperData[1];
            displayPaperDetails(paperId, paperDetails);
        }
    });
}

async function onKeyListener(event){
    if(event.keyCode == 13){
        searchQuery = document.getElementById('search-input').value;
        displayLoadingAnimation("background");
        await displaySearchResults(searchQuery);
    }
}