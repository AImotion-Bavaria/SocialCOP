
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>



[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h1 align="center">SocialCOP</h1>

  <p align="center">
    Most optimization algorithms aim to allocate limited resources among a set of agents. If human stakeholders are benefitting or suffering from this allocation, it is required to add a social perspective to the goal of efficient optimized allocation. SocialCOP introduces a library to integrate social choice concepts to a variety of existing MiniZinc optimization problems.
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#mapping">Mapping</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Many real-world combinatorial optimization problems dealing with the sharing of constrained resources involve multiple objectives belonging to their respective stakeholders. Examples thereof include shared mobility (carpooling, organizing collective trips, choosing which sites to visit and in what order), hoteling systems for shared office space (allocating desks to employees, respecting WFH restrictions or making joint reservations), shared manufacturing, or managing the charging of electric vehicles (EVs) in a shared parking lot. We call them collective combinatorial optimization problems. Aggregating stakeholders’ goals in a socially desirable way (e.g., fairness, proportional access, respect for democratic majority) while balancing this with other overarching goals is challenging. We therefore introduce a reusable library of constraints (e.g., envy-freeness or utilitarianism) and search procedures derived from social choice and fair division theory for a generic modeling language such as e.g. MiniZinc.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Please follow this guideline to install SocialCOP and use the different social choice concept in your existing MiniZinc problem.

### Prerequisites
Please install MiniZinc-python with the DZN flag activated
* MiniZinc
  
    ```sh
    pip install minizinc[dzn]
    ```
  

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/AImotion-Bavaria/SocialCOP.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Mapping
To connect your MiniZinc problem to the social choice building blocks, a social mapping file is required. It can be expressed using a JSON file. This file is required to map the fixed concepts to the respective names in your model formulation at hand. Required hooks like the array of agents taking part in the distribution, an array of utility decision variables, as well as the names of share functions and share utilities can therefore be named according to the underlying problem and do not need to be the same for all models. 

Example:
```{
 "agents_array" : "Agents",
 "utility_array" : "utilities",
 "num_agents" : "n_agents",
 "main_variables": ["assigned"]
 }
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Runners
#### Envy-Freeness
Envy-Freeness is given if no agent envies the distribution of another agent. To integrate it into your existing MiniZinc model, you can use the following code.
```def envy_free(model : Model, social_mapping : dict, solver : Solver):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
    simple_runner = prepare_envy_free_runner(social_mapping)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result 
```
#### Rawls Runner
Rawls runner is searching for the best result for the worst-off agent. To integrate it into your existing MiniZinc model, you can use the following code.
```def rawls(model: Model, social_mapping, solver: Solver):
    simple_runner = prepare_rawls_runner(social_mapping)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result
```

#### Nash
The Nash welfare is searching for the highest product of utilities. To integrate it into your existing MiniZinc model, you can use the following code.
```def nash(model : Model, social_mapping : dict, solver : Solver):
    simple_runner = prepare_nash_runner(social_mapping)
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result
```
#### Leximin
This runner is iteratively searching for rawls solution for all agents. The worst-off agent is maximized, followed by the second-worst agent in the next iteration. This is continued until all utilities for agents are optimized. To integrate it into your existing MiniZinc model, you can use the following code.
```def leximin(model: Model, social_mapping, solver: Solver):
    simple_runner = prepare_leximin_runner(social_mapping)
    simple_runner.debug = True
    simple_runner.debug_dir = create_debug_folder(os.path.dirname(__file__))
    simple_runner.timeout = TIME_LIMIT_EVAL
    result = simple_runner.run(model, solver)
    return result
```
#### Utilitarian Runner
Utilitarianism is searching for the result with the maximum overall utility. To integrate it into your existing MiniZinc model, you can use the following code.
```def utilitarian(model : Model, social_mapping : dict, solver : Solver):
    simple_runner = prepare_utilitarian_runner(social_mapping)
    if SHARE_FUNCTION in social_mapping: # it is a division problem - I want to record envy counts as well
        simple_runner.add(envy_freeness_mixin)
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(model, solver)
    return result
```
#### Combination  of social choice concepts
If you want to combine different social choice concepts e.g. Utilitarianism and Envy-Freeness you can implement it as follows:
```def utilitarian_envy_free(model : Model, social_mapping : dict, solver : Solver):
    if not SHARE_FUNCTION in social_mapping: # it is not  a division problem
        return None 
     
    simple_runner : SimpleRunner = prepare_utilitarian_runner(social_mapping)
    simple_runner.model += [envy_freeness_mixin, enforce_envy_freeness]
    simple_runner.timeout = TIME_LIMIT_EVAL

    result = simple_runner.run(model, solver)
    return result
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap


See the [open issues](https://github.com/AImotion-Bavaria/SocialCOP/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alexander Schiendorfer - Alexander.Schiendorfer@thi.de

Julia Ruttmann - jur9766@thi.de

Project Link: [https://github.com/AImotion-Bavaria/SocialCOP](https://github.com/AImotion-Bavaria/SocialCOP)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []() This project is being carried out within the Master of Applied Research in Engineering Sciences program of Technische Hochschule Ingolstadt.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/AImotion-Bavaria/SocialCOP.svg?style=for-the-badge
[contributors-url]: https://github.com/AImotion-Bavaria/SocialCOP/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/AImotion-Bavaria/SocialCOP.svg?style=for-the-badge
[forks-url]: https://github.com/AImotion-Bavaria/SocialCOP/network/members
[stars-shield]: https://img.shields.io/github/stars/AImotion-Bavaria/SocialCOP.svg?style=for-the-badge
[stars-url]: https://github.com/AImotion-Bavaria/SocialCOP/stargazers
[issues-shield]: https://img.shields.io/github/issues/AImotion-Bavaria/SocialCOP.svg?style=for-the-badge
[issues-url]: https://github.com/AImotion-Bavaria/SocialCOP/issues
[license-shield]: https://img.shields.io/github/license/AImotion-Bavaria/SocialCOP.svg?style=for-the-badge
[license-url]: https://github.com/AImotion-Bavaria/SocialCOP/blob/master/LICENSE.txt
