# Advanced_Recommendation_Systems_on_Google_Cloud

**Recommendation Systems Overview:**

Introduction

Recommendation systems

1. Help users find related content.
2. Help users explore new items.
3. Improve user decision-making.

For producers, Recommendation systems

1. Increase user engagement.
2. Learn more about customers.
3. Change user behavior.

Types of Recommendation Systems: 

User-item interaction matrix

Types:

1. Content-Based Recommender Systems
2. Collaborative  Filtering 
3. Knowledge-Based
4. Deep Learning is used to develop a recommendation model

Content-based Filtering or Collaborative Filtering:

Content-based Filtering uses item features to recommend new items similar to what the user has liked  in the past

Collaborative Filtering uses similarities between users and items simultaneously to determine recommendations.

CBF requires hand engineering the features but CF learns automatically as it recommends an item to user A based on the interest of another similar user B

CF involves matrix factorization and behaves  similarly to a CBF but doesn't rely on previously constructed features

Knowledge-based recommender systems use explicit knowledge about the users, items, and recommendation criteria.

Using these we can also build a hybrid model

Recommendation Systems Pitfalls

In the User-item interaction matrix the user space and the product space are sparse and skewed - sparse matrices are problematic cause they take up a lot of memory n r slow to perform computations - the matrix is skewed as some properties might be very popular 

The **cold start problem** occurs when there arent enough interactions for users or items.

If explicit information is not reliable or enough we can implicit information/feedback

Recommendation Systems Overview - Readings

[2 - Recommendation Systems Overview - Readings.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/f201c5d0-6e38-4a5b-9372-5a26fa1b351a/a8d46cdf-1021-4915-ac07-99b7210a8152/2_-_Recommendation_Systems_Overview_-_Readings.pdf)

Introduction | Recommendation Systems Overview

***Background***

**Terminology**

`https://developers.google.com/machine-learning/recommendation/overview/terminology#items-also-known-as-documents`

The entities a system recommends. For the Google Play store, the items are apps to install. For YouTube, the items are videos.

`https://developers.google.com/machine-learning/recommendation/overview/terminology#query-also-known-as-context`

The information a system uses to make recommendations. Queries can be a combination of the following:

- user information
    - the id of the user
    - items that users previously interacted with
- additional context
    - time of day
    - the user's device

**Recommendation Systems Overview**

One common architecture for recommendation systems consists of the following components:

- candidate generation
- scoring
- re-ranking


![30](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/7704bd62-b3de-4c80-950b-342f946fefa4)


`https://developers.google.com/machine-learning/recommendation/overview/types#candidate-generation`

In this first stage, the system starts from a potentially huge corpus and generates a much smaller subset of candidates. For example, the candidate generator in YouTube reduces billions of videos down to hundreds or thousands. The model needs to evaluate queries quickly given the enormous size of the corpus. A given model may provide multiple candidate generators, each nominating a different subset of candidates.

`https://developers.google.com/machine-learning/recommendation/overview/types#scoring`

Next, another model scores and ranks the candidates in order to select the set of items (on the order of 10) to display to the user. Since this model evaluates a relatively small subset of items, the system can use a more precise model relying on additional queries.

`https://developers.google.com/machine-learning/recommendation/overview/types#re-ranking`

Finally, the system must take into account additional constraints for the final ranking. For example, the system removes items that the user explicitly disliked or boosts the score of fresher content. Re-ranking can also help ensure diversity, freshness, and fairness.

***Candidate Generation***

**Candidate Generation Overview**

*Candidate generation is the first stage of recommendation. Given a query, the system generates a set of relevant candidates.* The following table shows two common candidate generation approaches:

| Type | Definition | Example |
| --- | --- | --- |
| content-based filtering | Uses similarity between items to recommend items similar to what the user likes. | If user A watches two cute cat videos, then the system can recommend cute animal videos to that user. |
| collaborative filtering | Uses similarities between queries and items simultaneously to provide recommendations. | If user A is similar to user B, and user B likes video 1, then the system can recommend video 1 to user A (even if user A hasn’t seen any videos similar to video 1). |

`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#embedding-space`

Both content-based and collaborative filtering map each item and each query (or context) to an embedding vector in a common embedding space *E=R^d*. Typically, the embedding space is low-dimensional (that is, *d* is much smaller than the size of the corpus), and captures some latent structure of the item or query set. Similar items, such as YouTube videos that are usually watched by the same user, end up close together in the embedding space. The notion of "closeness" is defined by a similarity measure.

http://projector.tensorflow.org/

`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#similarity-measures`

A similarity measure is a function $s : E \times E \to \mathbb R$ that takes a pair of embeddings and returns a scalar measuring their similarity. The embeddings can be used for candidate generation as follows: given a query embedding $q \in E$, the system looks for item embeddings $x \in E$ that are close to $q$, that is, embeddings with high similarity $s(q, x)$.

To determine the degree of similarity, most recommendation systems rely on one or more of the following:

- cosine
- dot product
- Euclidean distance

`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#cosine`

This is simply the cosine of the angle between the two vectors, $s(q, x) = \cos(q, x)$

`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#dot-product`

The dot product of two vectors is $s(q, x) = \langle q, x \rangle = \sum_{i = 1}^d q_i x_i$. It is also given by $s(q, x) = \|x\| \|q\| \cos(q, x)$ (the cosine of the angle multiplied by the product of norms). Thus, if the embeddings are normalized, then dot-product and cosine coincide.

`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#euclidean-distance`

This is the usual distance in Euclidean space, $s(q, x) = \|q - x\| = \left[ \sum_{i = 1}^d (q_i - x_i)^2\right]^{\frac{1}{2}}$. A smaller distance means higher similarity. Note that when the embeddings are normalized, the squared Euclidean distance coincides with dot-product (and cosine) up to a constant, since in that case $\frac{1}{2}\|q - x\|^2 = 1 - \langle q, x \rangle$.

`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#comparing-similarity-measures`

Consider the example in the figure to the right. The black vector illustrates the query embedding. The other three embedding vectors (Item A, Item B, Item C) represent candidate items. Depending on the similarity measure used, the ranking of the items can be different.

Using the image, try to determine the item ranking using all three of the similarity measures: cosine, dot product, and Euclidean distance.


![29](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/e0d7c31d-cec4-4f5f-8082-34ca067b9bac)


`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#expandable-1`

Answer - Item A has the largest norm, and is ranked higher according to the dot-product. Item C has the smallest angle with the query, and is thus ranked first according to the cosine similarity. Item B is physically closest to the query so Euclidean distance favors it.


![28](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/d5fc0583-af4c-44b5-9db9-d690ce716264)


`https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#which-similarity-measure-to-choose`

Compared to the cosine, the dot product similarity is sensitive to the norm of the embedding. That is, the larger the norm of an embedding, the higher the similarity (for items with an acute angle) and the more likely the item is to be recommended. This can affect recommendations as follows:

- Items that appear very frequently in the training set (for example, popular YouTube videos) tend to have embeddings with large norms. If capturing popularity information is desirable, then you should prefer dot product. However, if you're not careful, the popular items may end up dominating the recommendations. In practice, you can use other variants of similarity measures that put less emphasis on the norm of the item. For example, define $s(q, x) = \|q\|^\alpha \|x\|^\alpha \cos(q, x)$ for some $\alpha \in (0, 1)$.
- Items that appear very rarely may not be updated frequently during training. Consequently, if they are initialized with a large norm, the system may recommend rare items over more relevant items. To avoid this problem, be careful about embedding initialization, and use appropriate regularization. We will detail this problem in the first exercise.

**Content-based Filtering**

*Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.*

To demonstrate content-based filtering, let’s hand-engineer some features for the Google Play store. The following figure shows a feature matrix where each row represents an app and each column represents a feature. Features could include categories (such as Education, Casual, Health), the publisher of the app, and many others. To simplify, assume this feature matrix is binary: a non-zero value means the app has that feature.

You also represent the user in the same feature space. Some of the user-related features could be explicitly provided by the user. For example, a user selects "Entertainment apps" in their profile. Other features can be implicit, based on the apps they have previously installed. For example, the user installed another app published by Science R Us.

The model should recommend items relevant to this user. To do so, you must first pick a similarity metric (for example, dot product). Then, you must set up the system to score each candidate item according to this similarity metric. Note that the recommendations are specific to this user, as the model did not use any information about other users.

![27](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/55cda109-3c47-4b8e-bbc4-f72b860e1aa5)


`https://developers.google.com/machine-learning/recommendation/content-based/basics#using-dot-product-as-a-similarity-measure`

Consider the case where the user embedding $x$ and the app embedding $y$ are both binary vectors. Since $\langle x, y \rangle = \sum_{i = 1}^d x_i y_i$, a feature appears in both $x$ and $y$ contributes a 1 to the sum. In other words, $\langle x, y \rangle$ is the number of features that are active in both vectors simultaneously. A high dot product then indicates more common features, thus a higher similarity.

**Advantages & Disadvantages**

**Advantages**

- The model doesn't need any data about other users, since the recommendations are specific to this user. This makes it easier to scale to a large number of users.
- The model can capture the specific interests of a user, and can recommend niche items that very few other users are interested in.

**Disadvantages**

- Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge. Therefore, the model can only be as good as the hand-engineered features.
- The model can only make recommendations based on existing interests of the user. In other words, the model has limited ability to expand on the users' existing interests. 

**Collaborative Filtering and Matrix Factorization**

**Collaborative Filtering**

To address some of the limitations of content-based filtering, collaborative filtering uses *similarities between users and items simultaneously* to provide recommendations. This allows for serendipitous recommendations; that is, collaborative filtering models can recommend an item to user A based on the interests of a similar user B. Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features.

`https://developers.google.com/machine-learning/recommendation/collaborative/basics#a-movie-recommendation-example`

Consider a movie recommendation system in which the training data consists of a feedback matrix in which:

- Each row represents a user.
- Each column represents an item (a movie).

The feedback about movies falls into one of two categories:

- **Explicit**— users specify how much they liked a particular movie by providing a numerical rating.
- **Implicit**— if a user watches a movie, the system infers that the user is interested.

To simplify, we will assume that the feedback matrix is binary; that is, a value of 1 indicates interest in the movie.

When a user visits the homepage, the system should recommend movies based on both:

- similarity to movies the user has liked in the past
- movies that similar users liked

For the sake of illustration, let's hand-engineer some features for the movies described in the following table:

| Movie | Rating | Description |
| --- | --- | --- |
| The Dark Knight Rises | PG-13 | Batman endeavors to save Gotham City from nuclear annihilation in this sequel to The Dark Knight, set in the DC Comics universe. |
| Harry Potter and the Sorcerer's Stone | PG | A orphaned boy discovers he is a wizard and enrolls in Hogwarts School of Witchcraft and Wizardry, where he wages his first battle against the evil Lord Voldemort. |
| Shrek | PG | A lovable ogre and his donkey sidekick set off on a mission to rescue Princess Fiona, who is emprisoned in her castle by a dragon. |
| The Triplet of Belleville | PG-13 | When professional cycler Champion is kidnapped during the Tour de France, his grandmother and overweight dog journey overseas to rescue him, with the help of a trio of elderly jazz singers. |
| Memento | R | An amnesiac desperately seeks to solve his wife's murder by tattooing clues onto his body. |

`https://developers.google.com/machine-learning/recommendation/collaborative/basics#1d-embedding`

Suppose we assign to each movie a scalar in [−1,1] that describes whether the movie is for children (negative values) or adults (positive values). Suppose we also assign a scalar to each user in [−1,1] that describes the user's interest in children's movies (closer to -1) or adult movies (closer to +1). The product of the movie embedding and the user embedding should be higher (closer to 1) for movies that we expect the user to like.


![26](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/c30515fa-ac7a-4805-b7a8-0fd164aaada4)


In the diagram below, each checkmark identifies a movie that a particular user watched. The third and fourth users have preferences that are well explained by this feature—the third user prefers movies for children and the fourth user prefers movies for adults. However, the first and second users' preferences are not well explained by this single feature.


![25](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/c981f13f-f837-4a82-aadd-1886048eb2b5)


`https://developers.google.com/machine-learning/recommendation/collaborative/basics#2d-embedding`

One feature was not enough to explain the preferences of all users. To overcome this problem, let's add a second feature: the degree to which each movie is a blockbuster or an arthouse movie. With a second feature, we can now represent each movie with the following two-dimensional embedding:


![24](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/5a3d3841-12a2-4d11-a802-b29cd3ff6554)


We again place our users in the same embedding space to best explain the feedback matrix: for each (user, item) pair, we would like the dot product of the user embedding and the item embedding to be close to 1 when the user watched the movie, and to 0 otherwise.


![23](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/284669aa-6149-4664-9c1c-1c26f76a1d7a)


In this example, we hand-engineered the embeddings. In practice, the embeddings can be learned *automatically*, which is the power of collaborative filtering models. In the next two sections, we will discuss different models to learn these embeddings, and how to train them.

The collaborative nature of this approach is apparent when the model learns the embeddings. Suppose the embedding vectors for the movies are fixed. Then, the model can learn an embedding vector for the users to best explain their preferences. Consequently, embeddings of users with similar preferences will be close together. Similarly, if the embeddings for the users are fixed, then we can learn movie embeddings to best explain the feedback matrix. As a result, embeddings of movies liked by similar users will be close in the embedding space.

**Matrix Factorization**

Matrix factorization is a simple embedding model. Given the feedback matrix A $\in R^{m \times n}$, where  $m$  is the number of users (or queries) and $n$ is the number of items, the model learns:

- A user embedding matrix $U \in \mathbb R^{m \times d}$, where row i is the embedding for user i.
- An item embedding matrix $V \in \mathbb R^{n \times d}$, where row j is the embedding for item j.

![22](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/d782f153-e6ea-4717-aef4-a6a44076f8e9)


`https://developers.google.com/machine-learning/recommendation/collaborative/matrix#choosing-the-objective-function`

One intuitive objective function is the squared distance. To do this, minimize the sum of squared errors over all pairs of observed entries:

$\min_{U \in \mathbb R^{m \times d},\ V \in \mathbb R^{n \times d}} \sum_{(i, j) \in \text{obs}} (A_{ij} - \langle U_{i}, V_{j} \rangle)^2.$

![21](https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/5469a230-2dfa-4b1f-9c2e-6e3b55ea3925)


Perhaps you could treat the unobserved values as zero, and sum over all entries in the matrix. This corresponds to minimizing the squared [Frobenius](https://wikipedia.org/wiki/Matrix_norm#Frobenius_norm) distance between $A$ and its approximation $U V^T$:

$\min_{U \in \mathbb R^{m \times d},\ V \in \mathbb R^{n \times d}} \|A - U V^T\|_F^2.$

You can solve this quadratic problem through **Singular Value Decomposition** (**SVD**) of the matrix. However, SVD is not a great solution either, because in real applications, the matrix $A$ may be very sparse. For example, think of all the videos on YouTube compared to all the videos a particular user has viewed. The solution $UV^T$ (which corresponds to the model's approximation of the input matrix) will likely be close to zero, leading to poor generalization performance.

In contrast, **Weighted Matrix Factorization** decomposes the objective into the following two sums:

- A sum over observed entries.
- A sum over unobserved entries (treated as zeroes).

$\min_{U \in \mathbb R^{m \times d},\ V \in \mathbb R^{n \times d}} \sum_{(i, j) \in \text{obs}} (A_{ij} - \langle U_{i}, V_{j} \rangle)^2 + w_0 \sum_{(i, j) \not \in \text{obs}} (\langle U_i, V_j\rangle)^2.$

Here, $w_0$ is a hyperparameter that weights the two terms so that the objective is not dominated by one or the other. Tuning this hyperparameter is very important.

$\sum_{(i, j) \in \text{obs}} w_{i, j} (A_{i, j} - \langle U_i, V_j \rangle)^2 + w_0 \sum_{i, j \not \in \text{obs}} \langle U_i, V_j \rangle^2$

where $w_{i, j}$ is a function of the frequency of query i and item j.

`https://developers.google.com/machine-learning/recommendation/collaborative/matrix#minimizing-the-objective-function`

Common algorithms to minimize the objective function include:

- **[Stochastic gradient descent (SGD)](https://developers.google.com/machine-learning/crash-course/glossary#SGD)** is a generic method to minimize loss functions.
- **Weighted Alternating Least Squares** (**WALS**) is specialized to this particular objective.

The objective is quadratic in each of the two matrices U and V. (Note, however, that the problem is not jointly convex.) WALS works by initializing the embeddings randomly, then alternating between:

- Fixing  $U$  and solving for $V$ .
- Fixing $V$ and solving for $U$ .

Each stage can be solved exactly (via solution of a linear system) and can be distributed. This technique is guaranteed to converge because each step is guaranteed to decrease the loss.

`https://developers.google.com/machine-learning/recommendation/collaborative/matrix#sgd-vs.-wals`

SGD and WALS have advantages and disadvantages. Review the information below to see how they compare:

**SGD**

**Very flexible—can use other loss functions.**

**Can be parallelized.**

**Slower—does not converge as quickly.**

**Harder to handle the unobserved entries (need to use negative sampling or gravity).**

**WALS**

**Reliant on Loss Squares only.**

**Can be parallelized.**

**Converges faster than SGD.**

**Easier to handle unobserved entries.**

**Collaborative Filtering Advantages & Disadvantages**

`https://developers.google.com/machine-learning/recommendation/collaborative/summary#advantages`

**No domain knowledge necessary**

We don't need domain knowledge because the embeddings are automatically learned.

**Serendipity**

The model can help users discover new interests. In isolation, the ML system may not know the user is interested in a given item, but the model might still recommend it because similar users are interested in that item.

**Great starting point**

To some extent, the system needs only the feedback matrix to train a matrix factorization model. In particular, the system doesn't need contextual features. In practice, this can be used as one of multiple candidate generators.

`https://developers.google.com/machine-learning/recommendation/collaborative/summary#disadvantages`

**Cannot handle fresh items**

The prediction of the model for a given (user, item) pair is the dot product of the corresponding embeddings. So, if an item is not seen during training, the system can't create an embedding for it and can't query the model with this item. This issue is often called the **cold-start problem**. However, the following techniques can address the cold-start problem to some extent:

- **Projection in WALS.** Given a new item $i_0$ not seen in training, if the system has a few interactions with users, then the system can easily compute an embedding $v_{i_0}$ for this item without having to retrain the whole model. The system simply has to solve the following equation or the weighted version:
    
    $\min_{v_{i_0} \in \mathbb R^d} \|A_{i_0} - U v_{i_0}\|$
    
    The preceding equation corresponds to one iteration in WALS: the user embeddings are kept fixed, and the system solves for the embedding of item $i_0$. The same can be done for a new user.
    
- **Heuristics to generate embeddings of fresh items.** If the system does not have interactions, the system can approximate its embedding by averaging the embeddings of items from the same category, from the same uploader (in YouTube), and so on.

**Hard to include side features for query/item**

**Side features** are any features beyond the query or item ID. For movie recommendations, the side features might include country or age. Including available side features improves the quality of the model. Although it may not be easy to include side features in WALS, a generalization of WALS makes this possible.

To generalize WALS, **augment the input matrix with features** by defining a block matrix �¯, where:

- Block (0, 0) is the original feedback matrix $A$.
- Block (0, 1) is a multi-hot encoding of the user features.
- Block (1, 0) is a multi-hot encoding of the item features.

**Colab: Build a Movie Recommendation System**

https://colab.research.google.com/github/google/eng-edu/blob/main/ml/recommendation-systems/recommendation-systems.ipynb?utm_source=ss-recommendation-systems&utm_campaign=colab-external&utm_medium=referral&utm_content=recommendation-systems

**Recommendation Using Deep Neural Networks**

(Have to continue…)

**Retrieval, Scoring, and Re-ranking** 

(Have to continue…)

**Content-Based Recommendation Systems** 

Content-Based Recommendation Systems

Content-based filtering methods use item features to recommend new items that are similar to what the user has already liked based on their previous actions or explicit feedback they don't rely on information about other users or other user item interactions.

in this module we'll learn 

- how to measure the similarity of pairs of elements in an embedding space
- we'll discuss the mechanics of content-based recommendation systems and really see how they work
- we'll see how to build our own content-based recommendation systems one to recommend movies and one to recommend articles

Similarity Measures

In the quiz we had earlier we had some idea that the incredibles and shrek were similar but what does it mean for two movies to be similar we can try to answer this question in a number of ways one way is to consider the different genres or themes a movie has if there's a lot of overlap in themes between the two movies then perhaps we can say the movies are similar in the same way we can ask what it means for two users to be similar we could try to use the same genre analysis or another reasonable approach would be to consider the movies they've liked in the past if two users like a lot of the same movies we can say that those two users are similar to do machine learning we'll want to compare movies and users so we need to make this notion of similarity more rigorous this is often done by thinking of properties or features of items and users in the same embedding space where we can then compare how similar they are

An **embedding** is a map from our collection of items or users to some finite dimensional vector space it provides a way of giving a finite vector-valued representation of the items and users in our dataset 

embeddings are commonly used to represent input features in machine learning problems in fact we've already seen how embeddings can arise in some of the examples from previous modules here's a visualization of the mnist data set of handwritten images embedded into three dimensions if you look closely this embedding shows so all the ones lie in the same general area and all the twos lie together as well and so on

A **similarity measure** is just a metric that defines exactly how similar or close two items are in the embedding space 

one commonly used similarity measure is the **dot product** to compute the dot product of two vectors we compute the sum of the product-wise components of each vector and the **cosine similarity** is another popularly used similarity measure it's similar to the dot product but scaled by the norm.

**Building a User Vector**

let's go a bit deeper and look at a specific example putting some numbers to the previous ideas to see how exactly we can create a recommendation using content-based filtering 

consider a single user and suppose we have only seven movies in our database this user has seen and rated three of the movies we'd like to figure out which are the remaining four movies to recommend we'll assume a rating scale of one to ten one means they didn't like it and ten means they loved it this user gave shrek a 7 out of 10 blue a 4 out of 10 and harry potter a 10 out of 10.

we'd like to use this information to recommend one of the movies the user hasn't seen yet to do this we represent each of these movies using predetermined features or genres here we're using the genres fantasy action cartoon drama and comedy each movie is k-hot encoded as to whether it has that feature some movies satisfy only one feature some have more you can imagine with more granularity of features we'd be able to describe our movies in a more precise way but for now we'll just use these five categories given their previous movie ratings we can describe our user in terms of the same

<img width="683" alt="20" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/34dabc2a-e72b-433e-ac10-3fbec29b4350">


features we use to describe our movies that is we can place our user in the same 5-dimensional embedded feature space that we are using to represent our movies to do this we first scale each feature by this user's ratings and then normalize the resulting vector this is called the **user feature vector** basically it gives an idea of where our user sits in our embedding space of features based on their previous ratings of various movies in our database

<img width="681" alt="19" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/1863c68a-a400-4333-99e3-bc0be1812220">


let's work through that now first multiply the movie feature matrix by the ratings given by that user then aggregate by somewhere across each feature dimension this gives us a five dimensional vector in our feature space embedding the user feature vector is the normalization of that vector 

<img width="680" alt="18" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/17e347e7-bfeb-4209-85b6-f6b2ddac1583">


we see that for this user comedy seems to be a favorite category it has the largest value this makes sense looking back at their ratings for the three movies the two movies that were classified as comedy have relatively high ratings 7 out of 10 and 10 out of 10. The drama category appears to be the lowest which also makes sense looking at the rating of this user for the one drama movie they have seen they didn't rate it very highly the numeric values of the user feature vector make sense with the intuition we have from the user's ratings and the feature descriptions of the movies 

It is interesting to point out that the action dimension is zero for this user is this because the user doesn't like action at all not necessarily if you look at their ratings none of the movies they've previously rated contain the action feature think about how this affects our user feature vector.

**Making Recommendations Using a User Vector**

now we can make the best recommendation for our user based on their user feature vector and the features of the unrated unseen movies in our database to do this we'll use a similarity measure like we described previously basically we'll compute the dot product to measure the similarity between our user and all the remaining unranked movies in our database the movie with the greatest similarity measure is our top recommendation for our 

<img width="682" alt="17" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/14cf0be4-b97d-420a-a738-bee1e7909f40">


the dot product is found by taking the component wise product across each dimension and adding the results that is we multiply the user feature vector component wise with the movie feature vector for each movie and then sum row wise to compute the dot product this gives us the dot product similarity between our user and each ofthe four movies we'll use these values to make our recommendations

<img width="680" alt="16" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/111b62a1-6c8e-4ad0-bf1c-aae3adff7d11">


because star wars has the greatest similarity measure that will be our top recommendation followed by the incredibles and then the dark knight and lastly memento

**Making Recommendations for Many Users**

we've seen how content-based filtering can be used to generate movie recommendations for a single user we'd like to scale this technique so we can provide recommendations for multiple users at a time here we have a user movie rating matrix similar to the **user item interaction matrix** we saw in the previous module each row corresponds to a user and each column represents a movie in our database the value in row i and column j indicates the rating from 1 to 10 that user i gave movie j let's walk through the process we previously implemented to see how

this would look in theory and how you would implement this in tensorflow here is our user item rating matrix from earlier next to our **item feature matrix** remember the item feature matrix gives a **k hot encoding of the features** we are using to describe our movies each row corresponds to a single movie and a 1 indicates that the movie fits that genre

<img width="682" alt="15" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/5683aff0-a5b5-4ce8-9e34-eb7199f14f9a">


we can initialize these as constants in tensorflow by creating constant tensor values for our movies and for our movie features to do that we use tf constant to create two rank two tensors with the values for the user item rating matrix and the movie features from before hard coded.

<img width="512" alt="14" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/afa0c49e-5356-466f-b9b6-8cf7330b7353">


we want to **scale the movie feature matrix by their ratings given by each user** this will give us a **weighted feature matrix for each user** for the f*irst user we will get this weighted feature matrix we repeat this process for the second user and the third and so on* once we have this **collection of matrices we can stack them together using tf stack** to get a complete weighted user feature tensor

<img width="683" alt="13" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/c4c316f9-68cc-4237-b6a9-3fc19b01e07f">


the previous operations can be done in tensorflow in the following way we first build a list of the weighted feature matrices for each user then use tf stack applied to this list setting the stack axis to be 0.


<img width="503" alt="12" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/4abf3ed6-3280-4e50-9cec-af5e9fdaf421">


<img width="551" alt="11" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/5f1e276e-c135-4a74-a45e-54df5b545510">


To find the **user feature tensor** we sum across the feature columns just as before and individually normalize each of the resulting vectors

<img width="680" alt="10" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/768afa9e-fcdc-4e20-b4d6-d5af2de01670">


To normalize the user's movie features tensor we first sum each column using tf reduce sum and setting the axis equal to one the resulting tensor would then be rank two where each row represents the sum of the feature values for each user next we find the total for each user again using tf reduce sum with axis set to the normalization is then just the result of dividing the feature sum by the feature totals for each user in the end we stack the resulting tensors together to get the final user's features tensor

<img width="546" alt="9" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/f5a71d98-5d2d-4648-a0e9-7ef43cfaf781">


This results in a user feature tensor where each row corresponds to a specific user feature vector

<img width="562" alt="8" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/9e2df6e0-6600-46cd-9b85-d69ff689a31c">


To find the inferred movie rankings for our users we compute the dot product between each user feature vector and each movie feature vector in short we're seeing how similar each user is with respect to each movie as measured across these five feature dimensions for example for our first user the dot product with star wars gives 0.6 the dot product with the dark knight is 0.5 for shrek we get 0.4 and so on for the fourth movie and our last movie we do the same thing for user 2 for user 3 and finally user 4.


<img width="682" alt="7" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/1d6061bc-068a-45b5-8903-544f29802bd6">


to achieve this in tensorflow we can use the map function the tensorflow map function is similar to map function in numpy it repeatedly applies some callable function on a sequence of elements from first to last here we represent that function with lambda it is simply the dot product of each user feature vector with each movie feature vector so the variable user ratings holds a list of the resulting movie ratings for each user and each movie we stack them together to get a tensor of all the users and their movie ratings

<img width="545" alt="6" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/67ce27ef-5d4d-419c-b541-eb7ba5654278">


once we have the user movie ranking matrix we can compare it with the original user movie matrix to see which movies to recommend to which user because our users have already seen and rated some of our movies we want to mask the ranking for previously rated movies and focus only on unrated or unseen movies for each user.

<img width="586" alt="5" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/d56c297d-a4f0-462d-98f2-c85419cedae4">


<img width="275" alt="4" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/8592d2ba-b82d-4abb-9cf4-084a2758c1c1">


in the quiz we can accomplish this with tfware here the condition variable is a boolean and tfware will return either the tensor x or y depending on the value of the condition by setting a condition to be where the user item interaction matrix does not have any values we can return only those rankings for previously unrated movies this results in this user ranking matrix.

<img width="581" alt="3" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/53ef880b-1a15-41c3-89b6-459bc60c7ce2">

Finally, we can use the similarity rankings we've computed here to suggest new movies for each user for example for user 1 the incredibles has a higher similarity score than blue and so our new movie recommendation list for user one looks like this we can also do the same thing for all the other users.


<img width="563" alt="2" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/620b7693-0cec-47d6-be2b-d8ef0e782e8a">


before we move on let's take a closer look at our second user she has a rating of 0 for the dark knight why is that if we look at the original user movie rating matrix and compare with the movie feature matrix we can see why because this user has not rated anything containing action or sci-fi and the features of the dark knight are solely action and sci-fi our recommender system will infer a rating of 0 for that movie this actually highlights one of the drawbacks of content-based recommender systems it can be difficult to expand the interests of a user if the user hasn't rated movies within one of our predefined features our recommender won't suggest new movies in that genre in this sense content-based recommenders aren't good at expanding the interests of users to new domains

<img width="590" alt="1" src="https://github.com/chaitrasutari/Advanced_Recommendation_Systems_on_Google_Cloud/assets/83217099/cc1a6c67-7632-477f-8368-72ac75102617">


**Lab intro: Create a Content-Based Recommendation System**

now that we have learned the basic implementation of content-based filtering let's put our knowledge into action this slab demonstrates how to build a content-based recommender using tensorflow we'll put code to the multiple-user method we just discussed using most of the code we just saw you'll see how this can all be done using only low-level tensorflow operations.

**Implementing a Content-Based Filtering using Low Level TensorFlow Operations**

**Overview**

This lab shows you how to use low-level TensorFlow commands to do content-based filtering.

**Objectives**

In this lab, you learn how to perform the following tasks:

- Create and compute a user feature matrix.
- Compute where each user lies in the feature embedding space.
- Create recommendations for new movies based on similarity measures between the user and movie feature vectors.

**Introduction**

In this lab, you provide movie recommendations for a set of users. Content-based filtering uses features of the items and users to generate recommendations. In this small example, you use low-level TensorFlow operations and a very small set of movies and users to illustrate how this occurs in a larger content-based recommendation system.

**Task 1. Setup**

For each lab, you get a new Google Cloud project and set of resources for a fixed time at no cost.

1. Sign in to Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, `1:15:00`), and make sure you can finish within that time.
    
    There is no pause feature. You can restart if needed, but you have to start at the beginning.
    
3. When ready, click **Start lab**.
4. Note your lab credentials (**Username** and **Password**). You will use them to sign in to the Google Cloud Console.
5. Click **Open Google Console**.
6. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.
    
    If you use other credentials, you'll receive errors or **incur charges**.
    
7. Accept the terms and skip the recovery resource page.

**Note:** Do not click **End Lab** unless you have finished the lab or want to restart it. This clears your work and removes the project.

Enable the Vertex AI API

1. In the Google Cloud Console, on the **Navigation menu**, click **Vertex AI**.
2. Click **Enable Vertex AI API**.

Enable the Notebooks API

1. In the Google Cloud Console, on the **Navigation menu**, click **APIs & Services > Library**.
2. Search for **Notebooks API**, and press ENTER.
3. Click on the **Notebooks API** result.
4. If the API is not already enabled, click **Enable**.

**Task 2. Launch a Vertex AI Notebooks instance**

1. In the Google Cloud Console, on the **Navigation Menu**, click **Vertex AI > Workbench**. Select **User-Managed Notebooks**.
2. On the Notebook instances page, click **New Notebook > TensorFlow Enterprise > TensorFlow Enterprise 2.6 (with LTS) > Without GPUs**.
3. In the **New notebook** instance dialog, confirm the name of the deep learning VM, if you don’t want to change the region and zone, leave all settings as they are and then click **Create**. The new VM will take 2-3 minutes to start.
4. Click **Open JupyterLab**.
    
    A JupyterLab window will open in a new tab.
    
5. You will see “Build recommended” pop up, click **Build**. If you see the build failed, ignore it.

**Task 3. Clone a course repo within your Vertex AI Notebooks instance**

To clone the training-data-analyst notebook in your JupyterLab instance:

1. In JupyterLab, to open a new terminal, click the **Terminal** icon.
2. At the command-line prompt, run the following command:Copied!
    
    `git clone https://github.com/GoogleCloudPlatform/training-data-analyst`
    
    content_copy
    
3. To confirm that you have cloned the repository, double-click on the training-data-analyst directory and ensure that you can see its contents.
    
    The files for all the Jupyter notebook-based labs throughout this course are available in this directory.
    

**Task 4. Implement a content-based filtering using low level tensorflow operations**

1. In the notebook interface, navigate to **training-data-analyst > courses > machine_learning > deepdive2 > recommendation_systems > labs**, and open **content_based_by_hand.ipynb**.
2. In the notebook interface, click **Edit > Clear All Outputs**.
3. Carefully read through the notebook instructions, and fill in lines marked with #TODO where you need to complete the code.

**Tip:** To run the current cell, click the cell and press SHIFT+ENTER. Other cell commands are listed in the notebook UI under **Run**.

- Hints may also be provided for the tasks to guide you. Highlight the text to read the hints, which are in white text.
- To view the complete solution, navigate to **training-data-analyst > courses > machine_learning > deepdive2 > recommendation_systems > solutions**, and open **content_based_by_hand.ipynb**.

**End your lab**

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.

You will be given an opportunity to rate the lab experience. Select the applicable number of stars, type a comment, and then click **Submit**.

The number of stars indicates the following:

- 1 star = Very dissatisfied
- 2 stars = Dissatisfied
- 3 stars = Neutral
- 4 stars = Satisfied
- 5 stars = Very satisfied

You can close the dialog box if you don't want to provide feedback.

For feedback, suggestions, or corrections, please use the **Support** tab.

Copyright 2022 Google LLC All rights reserved. Google and the Google logo are trademarks of Google LLC. All other company and product names may be trademarks of the respective companies with which they are associated.

- **[Overview](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step1)**
- **[Objectives](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step2)**
- **[Introduction](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step3)**
- **[Task 1. Setup](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step4)**
- **[Task 2. Launch a Vertex AI Notebooks instance](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step5)**
- **[Task 3. Clone a course repo within your Vertex AI Notebooks instance](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step6)**
- **[Task 4. Implement a content-based filtering using low level tensorflow operations](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step7)**
- **[End your lab](https://www.cloudskillsboost.google/course_sessions/7203506/labs/325066#step8)**
