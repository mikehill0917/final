# Final
Michael Hill  
ISyE 6420  
May 1, 2022  

## Question 1
### Part 1
![beta_0](https://user-images.githubusercontent.com/22622059/166174211-3aa8cd30-a9a0-468f-b5f4-535998b09934.png)
![beta_1](https://user-images.githubusercontent.com/22622059/166174217-29e6da89-2ad4-49b9-b39f-64c1ab163b9b.png)
![beta_2](https://user-images.githubusercontent.com/22622059/166174222-8eb25521-c4b1-411d-92a2-e3c75a85d32b.png)
![var_e](https://user-images.githubusercontent.com/22622059/166174226-b0291ef6-a0e2-4090-80b8-edab82fa3c4e.png)
![var_u](https://user-images.githubusercontent.com/22622059/166174229-9efa9d2d-c584-4792-89e2-f144841db5d1.png)
### Part 2
![rho](https://user-images.githubusercontent.com/22622059/166174248-043c3d23-67fc-42d9-b075-4affb51c07cd.png)  
The posterior density of ρ appears to be significatly different from 0. This suggests that there is a significant intraclass correlation. 
In rough summary, each subject's orthodontic distance usually stays consistently ahead or behind the "usual" through this period of their childhood.
### Part 3
![beta_0](https://user-images.githubusercontent.com/22622059/166175852-556ace88-6309-49da-a95d-6a3eb510b1bc.png)
![beta_1](https://user-images.githubusercontent.com/22622059/166175853-e9de71f7-cb5e-4e81-b91e-41673a74ace0.png)
![beta_2](https://user-images.githubusercontent.com/22622059/166175856-fed671c8-3433-40a4-a390-b265369dddb3.png)
![var_e](https://user-images.githubusercontent.com/22622059/166175860-4c9671bf-ce51-45ca-b704-9711f4b3f0eb.png)  
Even though the medians of each of the beta parameters are similar to the random effects model, their confidence intervals are all a little wider. 
Without the additional information connecting observations of a single subject, we are less sure of our beta parameters. This is expecially true for Beta 2. 
This is likely because sex is consistent for each subject and the potency of its effect is now being spread between Beta 2 and the new random effect value.
The variance of the prediction nearly doubles. This suggests that adding the random effect value significantly increases the accuracy of the model.  
**Code:**
```
# Question 1
data = pd.read_csv("ortho.csv")

# Parts 1 and 2
with pm.Model() as model:
    # Define Priors
    beta = [pm.Normal(f"beta_{k}", mu=0, sigma=10**4) for k in [0, 1, 2]]
    tau_e = pm.Gamma("tau_e",  alpha=.01, beta=.01)
    tau_u = pm.Gamma("tau_u",  alpha=.01, beta=.01)
    # Define Models
    var_e = pm.Deterministic("var_e", 1 / tau_e)
    var_u = pm.Deterministic("var_u", 1 / tau_u)
    for subject in data["Subject"].unique():
        u_subject = pm.Normal(f"u_{subject}", mu=0, sigma=var_u**.5)
        subject_data = data.query("Subject == @subject")
        y_subject = pm.Normal(f"y_{subject}",
                              mu=beta[0] + beta[1]*subject_data["age"] + beta[2]*subject_data["Sex_coded"] + u_subject,
                              sigma=var_e**.5, observed=subject_data["y"])
    # Define Statistics
    rho = pm.Deterministic("rho", var_u / (var_u + var_e))
    # Trace
    trace = pm.sample(draws=100000, tune=10000, return_inferencedata=True, cores=1)
    # Plot Posteriors
    df = trace.posterior.to_dataframe()
    for column in df.columns:
        title = f"{column} median: {np.median(df[column])}\n" \
                f"95% CI: [{np.quantile(df[column], 0.025):.2f}, {np.quantile(df[column], 0.975):.2f}]"
        plt.figure()
        plt.hist(df[column], bins=50)
        plt.title(title)
        plt.savefig("part1/"+column+".png")

# Part 3
with pm.Model() as model:
    # Define Priors
    beta = [pm.Normal(f"beta_{k}", mu=0, sigma=10**4) for k in [0, 1, 2]]
    tau_e = pm.Gamma("tau_e",  alpha=.01, beta=.01)
    # Define Models
    var_e = pm.Deterministic("var_e", 1 / tau_e)
    y = pm.Normal("y", mu=beta[0] + beta[1]*data["age"] + beta[2]*data["Sex_coded"],
                  sigma=var_e**.5, observed=data["y"])
    # Trace
    trace = pm.sample(draws=100000, tune=10000, return_inferencedata=True, cores=1)
    # Plot Posteriors
    df = trace.posterior.to_dataframe()
    for column in df.columns:
        title = f"{column} median: {np.median(df[column])}\n" \
                f"95% CI: [{np.quantile(df[column], 0.025):.2f}, {np.quantile(df[column], 0.975):.2f}]"
        plt.figure()
        plt.hist(df[column], bins=50)
        plt.title(title)
        plt.savefig("part3/"+column+".png")
```
## Question 2
### Part 1
![theta_1](https://user-images.githubusercontent.com/22622059/166176617-6081e485-3099-41fa-be90-e7bf0cf711d3.png)
![theta_2](https://user-images.githubusercontent.com/22622059/166176620-8240c461-a863-4dd2-b673-476f950bfdc1.png)
![theta_3](https://user-images.githubusercontent.com/22622059/166176621-2099688d-6643-4f64-beb9-435e5f206586.png)
![theta_4](https://user-images.githubusercontent.com/22622059/166176625-3e858358-534a-4a1a-9bcb-c8d3f9d44e25.png)
### Part 2
![y_missing](https://user-images.githubusercontent.com/22622059/166176716-b6bc0d65-4100-4548-bba2-759312eaf07a.png)  
The funky pattern in this image is due to the interaction of the histogram's non-integer bin-width and the Poisson distribution's discrete nature.  
**Code:**
```
# Question 2
data = pd.read_csv("nanowire.csv").append({"x": 2.0}, ignore_index=True)

with pm.Model() as model:
    # Define Priors
    theta = {k: pm.Lognormal(f"theta_{k}", mu=0, sigma=10**.5) for k in [1, 3, 4]}
    theta[2] = pm.Uniform("theta_2", lower=0, upper=1)

    # Define Models
    mu = pm.Deterministic("mu", theta[1]*pm.math.exp(-theta[2]*(data["x"]**2)) +
                          theta[3]*(1-pm.math.exp(-theta[2]*(data["x"]**2)))*pm.invprobit(-np.array(data["x"])/theta[4]))
    y = pm.Poisson("y", mu=mu, observed=data["y"])
    # Trace
    trace = pm.sample(draws=100000, tune=10000, return_inferencedata=True, cores=1)
    # Plot Posteriors
    df = trace.posterior.to_dataframe()
    for column in df.columns:
        title = f"{column} median: {np.median(df[column])}\n" \
                f"95% CI: [{np.quantile(df[column], 0.025):.2f}, {np.quantile(df[column], 0.975):.2f}]"
        plt.figure()
        plt.hist(df[column], bins=50)
        plt.title(title)
        plt.savefig(column + ".png")
```

## Question 3
### Part a
Given the context of modeling counts of random events (bugs crawling onto the boards), I chose to model the dependent variable with a Poisson distribution.
I centered the prior of the general mean on the mean of the combined data. 
I centered the prior of each color's effect coefficent on the difference of that particular color's mean and the general mean.
The sigma of each prior was set to 100 because the majority of each color's counts fell within +-10 of its mean.
In order to statisfy the sum-to-zero constraint, I set the posterior of the "Lemon Yellow" coefficient to be the negated sum of the other color's coefficients.
To analyse the relative effects of each color, I also generated posterior distributions of the difference between each color pair.  
**Code:**
```
# Question 3
data = {"Lemon yellow": [45, 59, 48, 46, 38, 47],
            "White": [21, 12, 14, 17, 13, 17],
            "Green": [16, 11, 20, 21, 14, 7],
            "Blue": [37, 32, 15, 25, 39, 41]}


with pm.Model() as model:
    # Define Priors
    beta = {color: pm.Normal(f"beta_{color}", mu=np.mean(data[color])-np.mean(sum(data.values(), [])), sigma=100)
            for color in ["White", "Green", "Blue"]}
    # Define Models
    beta["Lemon yellow"] = pm.Deterministic(f"beta_Lemon yellow", -sum(beta.values()))  # Sum to Zero Constraint
    mu = pm.Normal("mu", mu=np.mean(sum(data.values(), [])), sigma=100)
    count = [pm.Poisson(f"count_{color}", mu=mu+beta[color], observed=data[color]) for color in data]
    # Comparisons
    color_diff = [pm.Deterministic(f"{color1}_minus_{color2}", beta[color1] - beta[color2])
                  for color1 in data for color2 in data if color1 != color2]
    # Trace
    trace = pm.sample(draws=100000, tune=10000, return_inferencedata=True, cores=1)
    # Plot Posteriors
    df = trace.posterior.to_dataframe()
    for column in df.columns:
        title = f"{column} median: {np.median(df[column])}\n" \
                f"95% CI: [{np.quantile(df[column], 0.025):.2f}, {np.quantile(df[column], 0.975):.2f}]"
        plt.figure()
        plt.hist(df[column], bins=50)
        plt.title(title)
        plt.savefig(column + ".png")
```
### Part b
Based on the posterior distributions shown below, I conclude that Lemon-Yellow is the most attractive color to these beetles and that Green and White are the least attractive color to these beetles. Blue lies somewhere in the middle of the "attractiveness spectrum".
The effect of each of these colors are significantly different from eachother except for Green and White which do not appear to have different levels of attractiveness from eachother.

![mu](https://user-images.githubusercontent.com/22622059/166176990-65e9df1c-c88a-4648-a0e4-cddaf92d4dc6.png)
![beta_Lemon yellow](https://user-images.githubusercontent.com/22622059/166177002-6fbc347e-dee2-468e-8d88-880bd9b300b1.png)
![beta_White](https://user-images.githubusercontent.com/22622059/166177014-571a385c-f6aa-4ca5-945e-e09b50ff2263.png)
![beta_Green](https://user-images.githubusercontent.com/22622059/166177018-44059a8a-6aac-4ac5-9f96-71a2c589ace5.png)
![beta_Blue](https://user-images.githubusercontent.com/22622059/166177021-a757e193-f3ad-43ba-a30d-6ade74c98bd9.png)
![Green_minus_White](https://user-images.githubusercontent.com/22622059/166177089-7c298942-a4da-4eb2-ac0e-9e9f67f51406.png)
