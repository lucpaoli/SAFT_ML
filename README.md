Code for Imperial Chemical Engineering 4th year research project. I need to clean it up badly!

The final results were generally extracted from
/jobs/24_long_scripts/
with a well-performing architecture being
7_att_clip_points_strat(x)

We ran the 5-fold cross validations manually by changing the fold selected on line 110.

To run the jobs, we submitted "run_job.sh" to the HPC cluster, which specified 100gb of ram and 16 cpus
100gb of ram was chosen because we occasionally ran out, though this appeared to be resolved after adding the manual call to the garbage collecter. However, if running this do keep an eye on that and consider tidying it up to use more inplace operations.

The important code is contained within "main.jl" for each folder, and "saftvrmienn.jl" in the root directory, which implements an AD-friendly version of saftvrmie, as implemented in Clapeyron.

Package versions shouldn't be a problem, other than noting that this worked with Clapeyron 0.5.7 and failed for 0.5.8. It's untested on 0.5.9.
