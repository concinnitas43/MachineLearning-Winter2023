\NeedsTeXFormat{LaTeX2e}

%%% Preamble — Packages
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass[11pt,a4paper]{article}
\usepackage{amsmath, amsthm, amssymb, amsfonts, amscd} % Useful math symbols and theorem environments
\RequirePackage{latexsym}
\RequirePackage{epsfig}
%\RequirePackage{graphics}
\RequirePackage{ifthen}
\RequirePackage{varioref}
\RequirePackage{color}
\RequirePackage{mathrsfs} % mathscr
\RequirePackage{tikz} % Drawing Diagrams
%\usepackage{kotex}
%\usepackage{siunitx}
%\RequirePackage[]{kotex}
%\usepackage{dhucs-untype1}
\RequirePackage{graphicx}

% Paper size
\RequirePackage[a4paper, total={5.5in, 9in}]{geometry}

% Font Related
% \usepackage{mlmodern} % Modern CMU
% \usepackage[T1]{fontenc}

%\usepackage{xeCJK}
%\usepackage{kotex}
%\setmainhangulfont{NanumMyeongjo}


\RequirePackage{titlesec} % Custom Section

%%% Theorem and Environments

\theoremstyle{definition}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{claim}[theorem]{Claim}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{question}{Question}
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\newtheorem{notation}[theorem]{Notation}
\newtheorem{exercise}[theorem]{Exercise}

%%% My Symbols
\renewcommand{\qed}{\hfill////}
\newcommand*{\dd}{\mathop{\mathrm{d}\!}}
\DeclareMathOperator{\Aut}{Aut}
\DeclareMathOperator{\Gal}{Gal}
\DeclareMathOperator{\Hom}{Hom}

%\setlength{\textwidth}{160.0mm}
%\setlength{\textheight}{240.0mm}
%\setlength{\oddsidemargin}{5mm}
%\setlength{\evensidemargin}{5mm}
%\setlength{\leftmargin}{50mm}
%\setlength{\topmargin}{-18mm}
%\setlength{\parindent}{5.0mm}
\setlength{\headheight}{14pt}
\addtolength{\topmargin}{-2pt}

% Section Bar
%\titleformat{\section}
  %{\normalfont\Large\bfseries}{\thesection}{1em}{}[{\vspace{0.3em}\titlerule[0.8pt]}]


