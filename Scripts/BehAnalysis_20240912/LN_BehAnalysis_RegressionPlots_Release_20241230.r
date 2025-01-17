# ------------------------------------------------------------------------------
# This script generates the regression plots for the linear mixed models used
# in the Garden Game behavioral study
# 
# Figures 3C-E, 5B-D, 6B, 7E and 8C-H
# 
# Laura Nett, 2024
# ------------------------------------------------------------------------------


# Packages
library(ggplot2)
library(lme4)
library(dplyr)
library(effects)
library(patchwork)

# Function for regression plots 3C, 5B-D and 6B (allocentric vs. egocentric performance for cohort1, cohort2 and cohorts1and2)
regression_plots_allo_ego <- function(predictor, xlab, breaks, xlim_range, outputfile, lineheight = 1, plotheight = c(10, 1)) {
    # Function to extract effects
    lmm <- function(cohort1_filtered, cohort2_filtered, data_complete, predictor) {
        # Formulas
        allocentric_formula <- as.formula(paste("AlloRetRankedPerformance ~", predictor, "+ (1|Subject)"))
        egocentric_formula  <- as.formula(paste("EgoRetRankedPerformance ~", predictor, "+ (1|Subject)"))

        # Fit models for Cohort 1
        m1_allocentric <- lmer(allocentric_formula, data = cohort1_filtered)
        m1_egocentric  <- lmer(egocentric_formula, data = cohort1_filtered)

        # Fit models for Cohort 2
        m2_allocentric <- lmer(allocentric_formula, data = cohort2_filtered)
        m2_egocentric  <- lmer(egocentric_formula, data = cohort2_filtered)

        # Fit models for combined Cohort 1 and 2
        m_comb_allocentric <- lmer(allocentric_formula, data = data_complete)
        m_comb_egocentric  <- lmer(egocentric_formula, data = data_complete)

        # Extract effects 
        eff1_allocentric <- as.data.frame(effect(predictor, m1_allocentric))
        eff1_egocentric  <- as.data.frame(effect(predictor, m1_egocentric))
        eff2_allocentric <- as.data.frame(effect(predictor, m2_allocentric))
        eff2_egocentric  <- as.data.frame(effect(predictor, m2_egocentric))
        eff_comb_allocentric <- as.data.frame(effect(predictor, m_comb_allocentric))
        eff_comb_egocentric  <- as.data.frame(effect(predictor, m_comb_egocentric))

        # Return the effects
        return(list(
            eff1_allocentric = eff1_allocentric,
            eff1_egocentric = eff1_egocentric,
            eff2_allocentric = eff2_allocentric,
            eff2_egocentric = eff2_egocentric,
            eff_comb_allocentric = eff_comb_allocentric,
            eff_comb_egocentric = eff_comb_egocentric
        ))
    }
    
    # Function to create one subplot
    create_plot <- function(eff_allo, eff_ego, predictor, breaks, ylab = NULL, show_yticks = TRUE, show_legend = FALSE, xlim_range = NULL) {
      # Convert the predictor name to a symbol
      predictor_sym <- sym(predictor)
        # Create plot
      p <- ggplot() +
        # Allocentric
        geom_line(data = eff_allo, aes(x = !!predictor_sym, y = fit, color = "Allocentric"), linewidth = 1.2) +
        geom_ribbon(data = eff_allo, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = "Allocentric"), alpha = 0.2) +
        
        # Egocentric
        geom_line(data = eff_ego, aes(x = !!predictor_sym, y = fit, color = "Egocentric"), linewidth = 1.2) +
        geom_ribbon(data = eff_ego, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = "Egocentric"), alpha = 0.2) +
        ylim(0.7, 1) + 
        scale_color_manual(values = c("Allocentric" = "#0000FF", "Egocentric" = "#008000")) +
        scale_fill_manual(values = c("Allocentric" = "#0000FF", "Egocentric" = "#008000"), guide = 'none') +
        scale_x_continuous(breaks = breaks, limits = xlim_range) +  # Dynamic x-axis limits 
        theme_minimal() +
        theme(
          legend.position = if (show_legend) c(0.45, 0.9) else "none",
          legend.title = element_blank(),
          legend.text = element_text(size = 22),
          axis.text.x = element_text(size = 24, color = "black"),
          axis.text.y = element_text(size = 24, color = "black"),
          axis.title.x = element_blank(),
          axis.title.y = element_text(size = 26),
          plot.margin = margin(1, 10, 1, 1),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
        ) +
        guides(color = guide_legend(override.aes = list(fill = NA)))

      if (!show_yticks) {
        p <- p + theme(axis.text.y = element_blank(), axis.title.y = element_blank())
      }

      if (!is.null(ylab)) {
        p <- p + labs(y = ylab)
      }

      return(p)
    }
    
    # Extract the effects
    eff <- lmm(cohort1_filtered, cohort2_filtered, data_complete, predictor)
    eff1_allocentric <- eff$eff1_allocentric
    eff1_egocentric  <- eff$eff1_egocentric
    eff2_allocentric <- eff$eff2_allocentric
    eff2_egocentric  <- eff$eff2_egocentric
    eff_comb_allocentric <- eff$eff_comb_allocentric
    eff_comb_egocentric  <- eff$eff_comb_egocentric

    # Cohort 1
    plot1 <- create_plot(eff1_allocentric, eff1_egocentric, predictor, ylab = "Performance", breaks, show_yticks = TRUE, show_legend = TRUE, xlim_range = xlim_range) +
      ggtitle("Cohort 1") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Cohort 2
    plot2 <- create_plot(eff2_allocentric, eff2_egocentric, predictor, ylab = NULL, breaks, show_yticks = FALSE, show_legend = FALSE, xlim_range = xlim_range) +
      ggtitle("Cohort 2") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Cohorts 1 and 2
    plot_comb <- create_plot(eff_comb_allocentric, eff_comb_egocentric, predictor, ylab = NULL, breaks, show_yticks = FALSE, show_legend = FALSE, xlim_range = xlim_range) +
      ggtitle("Cohorts 1 and 2") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))
        x_label_plot <- ggplot() +
          theme_void() +
          annotate("text", x = 0.5, y = 0.5, label = xlab, size = 9, vjust = 0.4, hjust = 0.5, lineheight = lineheight)

    # Combine the plots
    plots_combined <- plot1 + plot_spacer() + plot2 + plot_spacer() + plot_comb +
      plot_layout(widths = c(10, 0.1, 10, 0.1, 10))  

    # Add the x-axis label 
    final_plot <- (plots_combined / x_label_plot) + 
      plot_layout(heights = plotheight)  

    # Save the final plot 
    ggsave(
      filename = outputfile,
      plot = final_plot,     
      width = 9,            
      height = 6,           
      device = "svg"          
    )
    }

# Function for regression plots 3D+E (stable vs. unstable allocentric or egocentric performance for cohort1, cohort2 and cohorts1and2)
regression_plots_stable_unstable <- function(dependent_variable, predictor, xlab, breaks, color_stable, color_unstable, xlim_range, outputfile, lineheight = 1, plotheight = c(10, 1)) {
    # Function to extract effects
    lmm <- function(cohort1_stable, cohort1_unstable, cohort2_stable, cohort2_unstable, data_complete_stable, data_complete_unstable, dependent_variable, predictor) {
        # Formulas
        formula <- as.formula(paste(dependent_variable, "~", predictor, "+ (1|Subject)"))

        # Fit models for Cohort 1
        m1_stable <- lmer(formula, data = cohort1_stable)
        m1_unstable  <- lmer(formula, data = cohort1_unstable)

        # Fit models for Cohort 2
        m2_stable <- lmer(formula, data = cohort2_stable)
        m2_unstable  <- lmer(formula, data = cohort2_unstable)

        # Fit models for combined Cohort 1 and 2 
        m_comb_stable <- lmer(formula, data = data_complete_stable)
        m_comb_unstable  <- lmer(formula, data = data_complete_unstable)

        # Extract effects 
        eff1_stable <- as.data.frame(effect(predictor, m1_stable))
        eff1_unstable  <- as.data.frame(effect(predictor, m1_unstable))
        eff2_stable <- as.data.frame(effect(predictor, m2_stable))
        eff2_unstable  <- as.data.frame(effect(predictor, m2_unstable))
        eff_comb_stable <- as.data.frame(effect(predictor, m_comb_stable))
        eff_comb_unstable  <- as.data.frame(effect(predictor, m_comb_unstable))

        # Return the effects
        return(list(
            eff1_stable = eff1_stable,
            eff1_unstable = eff1_unstable,
            eff2_stable = eff2_stable,
            eff2_unstable = eff2_unstable,
            eff_comb_stable = eff_comb_stable,
            eff_comb_unstable = eff_comb_unstable
        ))
    }
    
    # Function to create one subplot
    create_plot <- function(eff_stable, eff_unstable, predictor, breaks, color_stable, color_unstable, ylab = NULL, show_yticks = TRUE, show_legend = FALSE, xlim_range = NULL) {
      # Convert the predictor name to a symbol
      predictor_sym <- sym(predictor)
        # Create plot
      p <- ggplot() +
        # Allocentric
        geom_line(data = eff_stable, aes(x = !!predictor_sym, y = fit, color = "Stable"), linewidth = 1.2) +
        geom_ribbon(data = eff_stable, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = "Stable"), alpha = 0.2) +
        
        # Egocentric
        geom_line(data = eff_unstable, aes(x = !!predictor_sym, y = fit, color = "Unstable"), linewidth = 1.2) +
        geom_ribbon(data = eff_unstable, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = "Unstable"), alpha = 0.2) +
        ylim(0.7, 1) +  # Set static y-axis limits
        scale_color_manual(values = c("Stable" = color_stable, "Unstable" = color_unstable)) +
        scale_fill_manual(values = c("Stable" = color_stable, "Unstable" = color_unstable), guide = 'none') +
        scale_x_continuous(breaks = breaks, limits = xlim_range) +  
        theme_minimal() +
        theme(
          legend.position = if (show_legend) c(0.4, 0.9) else "none",
          legend.title = element_blank(),
          legend.text = element_text(size = 22),
          axis.text.x = element_text(size = 24, color = "black"),
          axis.text.y = element_text(size = 24, color = "black"),
          axis.title.x = element_blank(),
          axis.title.y = element_text(size = 26),
          plot.margin = margin(1, 10, 1, 1),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
        ) +
        guides(color = guide_legend(override.aes = list(fill = NA)))

      if (!show_yticks) {
        p <- p + theme(axis.text.y = element_blank(), axis.title.y = element_blank())
      }

      if (!is.null(ylab)) {
        p <- p + labs(y = ylab)
      }

      return(p)
    }
    
    # Extract the effects
    eff <- lmm(cohort1_stable, cohort1_unstable, cohort2_stable, cohort2_unstable, data_complete_stable, data_complete_unstable, dependent_variable, predictor)
    eff1_stable <- eff$eff1_stable
    eff1_unstable  <- eff$eff1_unstable
    eff2_stable <- eff$eff2_stable
    eff2_unstable  <- eff$eff2_unstable
    eff_comb_stable <- eff$eff_comb_stable
    eff_comb_unstable  <- eff$eff_comb_unstable

    # Create plots
    ylab_plot <- if (dependent_variable == "AlloRetRankedPerformance") {
            "Allocentric performance"
        } else if (dependent_variable == "EgoRetRankedPerformance") {
            "Egocentric performance"
        } else {
            stop("Invalid dependent_variable Use 'AlloRetRankedPerformance' or 'EgoRetRankedPerformance'.")
        }
    
    # Cohort 1
    plot1 <- create_plot(eff1_stable, eff1_unstable, predictor, ylab = ylab_plot, breaks, color_stable, color_unstable, show_yticks = TRUE, show_legend = TRUE, xlim_range = xlim_range) +
      ggtitle("Cohort 1") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Cohort 2
    plot2 <- create_plot(eff2_stable, eff2_unstable, predictor, ylab = NULL, breaks, color_stable, color_unstable, show_yticks = FALSE, show_legend = FALSE, xlim_range = xlim_range) +
      ggtitle("Cohort 2") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Cohorts 1 and 2
    plot_comb <- create_plot(eff_comb_stable, eff_comb_unstable, predictor, ylab = NULL, breaks, color_stable, color_unstable, show_yticks = FALSE, show_legend = FALSE, xlim_range = xlim_range) +
      ggtitle("Cohorts 1 and 2") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))
        x_label_plot <- ggplot() +
          theme_void() +
          annotate("text", x = 0.5, y = 0.5, label = xlab, size = 9, vjust = 0.4, hjust = 0.5, lineheight = lineheight)

    # Combine the plots 
    plots_combined <- plot1 + plot_spacer() + plot2 + plot_spacer() + plot_comb +
      plot_layout(widths = c(10, 0.1, 10, 0.1, 10))  

    # Add the x-axis label
    final_plot <- (plots_combined / x_label_plot) + 
      plot_layout(heights = plotheight)  

    # Save the final plot 
    ggsave(
      filename = outputfile, 
      plot = final_plot,     
      width = 9,            
      height = 6,            
      device = "svg"            
    )
    }

# Function for regression figure 7E (cohort1 vs. cohort2 vs. cohorts1and2 for allocentric and egocentric)
regression_plots_7E <- function(predictor, xlab, breaks, xlim_range, outputfile, lineheight = 1, plotheight = c(10, 1)) {
    # Function to extract effects
    lmm <- function(cohort1_filtered, cohort2_filtered, data_complete, predictor) {
        # Formulas
        allocentric_formula <- as.formula(paste("AlloRetRankedPerformance ~", predictor, "+ (1|Subject)"))
        egocentric_formula  <- as.formula(paste("EgoRetRankedPerformance ~", predictor, "+ (1|Subject)"))

        # Fit models for Cohort 1
        m1_allocentric <- lmer(allocentric_formula, data = cohort1_filtered)
        m1_egocentric  <- lmer(egocentric_formula, data = cohort1_filtered)

        # Fit models for Cohort 2
        m2_allocentric <- lmer(allocentric_formula, data = cohort2_filtered)
        m2_egocentric  <- lmer(egocentric_formula, data = cohort2_filtered)

        # Fit models for combined Cohort 1 and 2 
        m_comb_allocentric <- lmer(allocentric_formula, data = data_complete)
        m_comb_egocentric  <- lmer(egocentric_formula, data = data_complete)

        # Extract effects 
        eff1_allocentric <- as.data.frame(effect(predictor, m1_allocentric))
        eff1_egocentric  <- as.data.frame(effect(predictor, m1_egocentric))
        eff2_allocentric <- as.data.frame(effect(predictor, m2_allocentric))
        eff2_egocentric  <- as.data.frame(effect(predictor, m2_egocentric))
        eff_comb_allocentric <- as.data.frame(effect(predictor, m_comb_allocentric))
        eff_comb_egocentric  <- as.data.frame(effect(predictor, m_comb_egocentric))

        # Return the effects
        return(list(
            eff1_allocentric = eff1_allocentric,
            eff1_egocentric = eff1_egocentric,
            eff2_allocentric = eff2_allocentric,
            eff2_egocentric = eff2_egocentric,
            eff_comb_allocentric = eff_comb_allocentric,
            eff_comb_egocentric = eff_comb_egocentric
        ))
    }
    
    # Function to create one subplot
    create_plot <- function(eff_cohort1, eff_cohort2, eff_cohorts, predictor, breaks, ylab = NULL, show_yticks = TRUE, show_legend = TRUE, xlim_range = NULL) {
        # Convert the predictor name to a symbol
      predictor_sym <- sym(predictor)
        # Create plot
      p <- ggplot() +
        # Cohort 1
        geom_line(data = eff_cohort1, aes(x = !!predictor_sym, y = fit, color = "Cohort 1"), linewidth = 1.2) +
        geom_ribbon(data = eff_cohort1, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = "Cohort 1"), alpha = 0.2, show.legend = FALSE) +
        
        # Cohort 2
        geom_line(data = eff_cohort2, aes(x = !!predictor_sym, y = fit, color = "Cohort 2"), linewidth = 1.2) +
        geom_ribbon(data = eff_cohort2, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = "Cohort 2"), alpha = 0.2, show.legend = FALSE) +
        
        # Cohorts 1 and 2
        geom_line(data = eff_cohorts, aes(x = !!predictor_sym, y = fit, color = "Cohorts 1 and 2"), linewidth = 1.2) +
        geom_ribbon(data = eff_cohorts, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = "Cohorts 1 and 2"), alpha = 0.2, show.legend = FALSE) +

        ylim(0.5, 1.05) + 
        scale_color_manual(values = c(
          "Cohort 1" = "#1f77b4",       
          "Cohort 2" = "#0056a3",      
          "Cohorts 1 and 2" = "#6baed6"
        )) +
        scale_fill_manual(values = c(
          "Cohort 1" = "#1f77b4",
          "Cohort 2" = "#0056a3",
          "Cohorts 1 and 2" = "#6baed6"
        )) +
        scale_x_continuous(breaks = breaks, limits = xlim_range) +  
        theme_minimal() +
        theme(
          legend.position = if (show_legend) c(0.5, 0.88) else "none",
          legend.title = element_blank(),
          legend.text = element_text(size = 19),
          axis.text.x = element_text(size = 24, color = "black"),
          axis.text.y = element_text(size = 24, color = "black"),
          axis.title.x = element_blank(),
          axis.title.y = element_text(size = 26),
          plot.margin = margin(1, 10, 1, 1),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
        ) +
        guides(color = guide_legend(override.aes = list(fill = NA))) 

      if (!show_yticks) {
        p <- p + theme(axis.text.y = element_blank(), axis.title.y = element_blank())
      }

      if (!is.null(ylab)) {
        p <- p + labs(y = ylab)
      }

      return(p)
    }
    
    # Updated color schemes for egcentric plot
    egocentric_colors <- c(
      "Cohort 1" = "#2ca02c",  
      "Cohort 2" = "#1b8c1b",  
      "Cohorts 1 and 2" = "#98df8a" 
    )
    
    # Extract the effects
    eff <- lmm(cohort1_filtered, cohort2_filtered, data_complete, predictor)
    eff1_allocentric <- eff$eff1_allocentric
    eff1_egocentric  <- eff$eff1_egocentric
    eff2_allocentric <- eff$eff2_allocentric
    eff2_egocentric  <- eff$eff2_egocentric
    eff_comb_allocentric <- eff$eff_comb_allocentric
    eff_comb_egocentric  <- eff$eff_comb_egocentric

    # Allocentric plot
    plot1 <- create_plot(eff1_allocentric, eff2_allocentric, eff_comb_allocentric, predictor, breaks,
                         ylab = "Performance", show_yticks = TRUE, show_legend = TRUE, xlim_range = xlim_range) +
      ggtitle("Allocentric") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Egocentric plot
    plot2 <- create_plot(eff1_egocentric, eff2_egocentric, eff_comb_egocentric, predictor, breaks,
                         ylab = NULL, show_yticks = FALSE, show_legend = TRUE, xlim_range = xlim_range) +
      scale_color_manual(values = egocentric_colors) + # Change colors
      scale_fill_manual(values = egocentric_colors) +  # Change colors
      ggtitle("Egocentric") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Combine the plots and add x-axis label
    x_label_plot <- ggplot() +
      theme_void() +
      annotate("text", x = 0.5, y = 0.5, label = xlab, size = 9, vjust = 0.4, hjust = 0.5, lineheight = lineheight)

    plots_combined <- plot1 + plot2  
    
    final_plot <- (plots_combined / x_label_plot) + 
      plot_layout(heights = plotheight)  

    # Save the final plot
    ggsave(
      filename = outputfile,
      plot = final_plot,
      width = 6,
      height = 6,
      device = "svg"
    )
    }


# Function for regression plots 8C&D (Eye tracking for one cohort, allo vs. ego with multiple predictors)
regression_plot_8CD <- function(predictors, xlab, breaks, xlim_range, outputfile, labels, colors_allocentric, colors_egocentric, linetypes, lineheight = 0.85, plotheight = c(10, 2)) {
    # Function to extract effects
    lmm <- function(cohort2_filtered, predictor) {
        # Formulas
        allocentric_formula <- as.formula(paste("AlloRetRankedPerformance ~", predictor, "+ (1|Subject)"))
        egocentric_formula  <- as.formula(paste("EgoRetRankedPerformance ~", predictor, "+ (1|Subject)"))

        # Fit models for Cohort 2
        m_allocentric <- lmer(allocentric_formula, data = cohort2_filtered)
        m_egocentric  <- lmer(egocentric_formula, data = cohort2_filtered)

        # Extract effects 
        eff_allocentric <- as.data.frame(effect(predictor, m_allocentric))
        eff_egocentric  <- as.data.frame(effect(predictor, m_egocentric))

        # Return the effects
        return(list(
            eff_allocentric = eff_allocentric,
            eff_egocentric = eff_egocentric
        ))
    }
    
    # Function to create one subplot
    create_plot <- function(eff_sub1, eff_sub2, eff_sub3, eff_sub4, eff_main, predictors, ylab = NULL, show_yticks = TRUE, show_legend = TRUE, xlim_range = NULL, colors, linetypes, labels) {
        # Convert the predictor names to symbols
        predictor_sym_sub1 <- sym(predictors[1])
        predictor_sym_sub2 <- sym(predictors[2])
        predictor_sym_sub3 <- sym(predictors[3])
        predictor_sym_sub4 <- sym(predictors[4])
        predictor_sym_main <- sym(predictors[5])
        
        # Create plot
        p <- ggplot() +    
        # Subset 1
        geom_line(data = eff_sub1, aes(x = !!predictor_sym_sub1, y = fit, color = labels$subsets[1], linetype = labels$subsets[1]), linewidth = 1.2) +
        geom_ribbon(data = eff_sub1, aes(x = !!predictor_sym_sub1, ymin = lower, ymax = upper, fill = labels$subsets[1]), alpha = 0.2, show.legend = FALSE) +

        # Subset 2
        geom_line(data = eff_sub2, aes(x = !!predictor_sym_sub2, y = fit, color = labels$subsets[2], linetype = labels$subsets[2]), linewidth = 1.2) +
        geom_ribbon(data = eff_sub2, aes(x = !!predictor_sym_sub2, ymin = lower, ymax = upper, fill = labels$subsets[2]), alpha = 0.2, show.legend = FALSE) +

        # Subset 3
        geom_line(data = eff_sub3, aes(x = !!predictor_sym_sub3, y = fit, color = labels$subsets[3], linetype = labels$subsets[3]), linewidth = 1.2) +
        geom_ribbon(data = eff_sub3, aes(x = !!predictor_sym_sub3, ymin = lower, ymax = upper, fill = labels$subsets[3]), alpha = 0.2, show.legend = FALSE) +

        # Subset 4
        geom_line(data = eff_sub4, aes(x = !!predictor_sym_sub4, y = fit, color = labels$subsets[4], linetype = labels$subsets[4]), linewidth = 1.2) +
        geom_ribbon(data = eff_sub4, aes(x = !!predictor_sym_sub4, ymin = lower, ymax = upper, fill = labels$subsets[4]), alpha = 0.2, show.legend = FALSE) +

        # Main 
        geom_line(data = eff_main, aes(x = !!predictor_sym_main, y = fit, color = labels$main, linetype = labels$main), linewidth = 1.2) +
        geom_ribbon(data = eff_main, aes(x = !!predictor_sym_main, ymin = lower, ymax = upper, fill = labels$main), alpha = 0.2, show.legend = FALSE) +

        ylim(0.55, 1.2) +  
        scale_color_manual(values = colors, breaks = c(labels$main, labels$subsets[1], labels$subsets[2], labels$subsets[3], labels$subsets[4])) +
        scale_fill_manual(values = colors) +
        scale_linetype_manual(values = linetypes, breaks = c(labels$main, labels$subsets[1], labels$subsets[2], labels$subsets[3], labels$subsets[4])) +
        scale_x_continuous(breaks = breaks, limits = xlim_range) +  # Dynamic x-axis limits
        scale_y_continuous(limits = c(0.55, 1.2),  # Set the y-axis range
                           breaks = c(0.6, 0.7, 0.8, 0.9, 1.0)  # Set the tick marks
        ) +
        theme_minimal() +
        theme(
          legend.position = if (show_legend) c(0.45, 0.82) else "none",
          legend.title = element_blank(),
          legend.text = element_text(size = 22),
          axis.text.x = element_text(size = 24, color = "black"),
          axis.text.y = element_text(size = 24, color = "black"),
          axis.title.x = element_blank(),
          axis.title.y = element_text(size = 26),
          plot.margin = margin(1, 10, 1, 1),
          panel.grid = element_blank(),
          panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
        ) +
        guides(
        color = guide_legend(override.aes = list(linetype = linetypes[labels$main])), 
        linetype = "none" 
        )

        if (!show_yticks) {
        p <- p + theme(axis.text.y = element_blank(), axis.title.y = element_blank())
        }

        if (!is.null(ylab)) {
        p <- p + labs(y = ylab)
        }

        return(p)
        }
    
    # Extract the effects
    eff_sub1 <- lmm(cohort2_filtered, predictors[1])
    eff_sub1_allo <- eff_sub1$eff_allocentric
    eff_sub1_ego <- eff_sub1$eff_egocentric
    eff_sub2 <- lmm(cohort2_filtered, predictors[2])
    eff_sub2_allo <- eff_sub2$eff_allocentric
    eff_sub2_ego <- eff_sub2$eff_egocentric
    eff_sub3 <- lmm(cohort2_filtered, predictors[3])
    eff_sub3_allo <- eff_sub3$eff_allocentric
    eff_sub3_ego <- eff_sub3$eff_egocentric
    eff_sub4 <- lmm(cohort2_filtered, predictors[4])
    eff_sub4_allo <- eff_sub4$eff_allocentric
    eff_sub4_ego <- eff_sub4$eff_egocentric
    eff_main <- lmm(cohort2_filtered, predictors[5])
    eff_main_allo <- eff_main$eff_allocentric
    eff_main_ego <- eff_main$eff_egocentric
    
    # Allocentric 
    plot1 <- create_plot(eff_sub1_allo, eff_sub2_allo, eff_sub3_allo, eff_sub4_allo, eff_main_allo, predictors,
                         ylab = "Performance", show_yticks = TRUE, show_legend = TRUE, xlim_range = xlim_range,
                         colors = colors_allocentric, linetypes = linetypes, labels = labels) +
                         ggtitle("Allocentric") +
                         theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Egocentric 
    plot2 <- create_plot(eff_sub1_ego, eff_sub2_ego, eff_sub3_ego, eff_sub4_ego, eff_main_ego, predictors,
                         ylab = NULL, show_yticks = FALSE, show_legend = TRUE, xlim_range = xlim_range,
                         colors = colors_egocentric, linetypes = linetypes, labels = labels) +
                         ggtitle("Egocentric") +
                         theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Combine the plots 
    x_label_plot <- ggplot() +
      theme_void() +
      annotate("text", x = 0.5, y = 0.5, label = xlab, size = 9, vjust = 0.4, hjust = 0.5, lineheight = lineheight)

    plots_combined <- plot1 + plot2  

    final_plot <- (plots_combined / x_label_plot) + 
      plot_layout(heights = plotheight)  

    # Save the final plot
    ggsave(
      filename = outputfile,
      plot = final_plot,
      width = 6,
      height = 6,
      device = "svg"
    )

    }

# Function for regression plots 8E-H (Eye tracking for one cohort, allo vs. ego)
regression_plots_eye <- function(predictor, xlab, breaks, xlim_range, outputfile, lineheight = 0.85, plotheight = c(10, 2)) {
    # Function to extract effects
    lmm <- function(cohort2_filtered, predictor) {
        # Formulas
        allocentric_formula <- as.formula(paste("AlloRetRankedPerformance ~", predictor, "+ (1|Subject)"))
        egocentric_formula  <- as.formula(paste("EgoRetRankedPerformance ~", predictor, "+ (1|Subject)"))

        # Fit models for Cohort 2
        m_allocentric <- lmer(allocentric_formula, data = cohort2_filtered)
        m_egocentric  <- lmer(egocentric_formula, data = cohort2_filtered)

        # Extract effects 
        eff_allocentric <- as.data.frame(effect(predictor, m_allocentric))
        eff_egocentric  <- as.data.frame(effect(predictor, m_egocentric))

        # Return the effects
        return(list(
            eff_allocentric = eff_allocentric,
            eff_egocentric = eff_egocentric
        ))
    }
    
    # Function to create one subplot
    create_plot <- function(eff, plot_type, predictor, breaks, ylab = NULL, show_yticks = TRUE, show_legend = FALSE, xlim_range = NULL) {
        # Set colors based on the plot type (Allocentric or Egocentric)
        color_map <- c("Allocentric" = "#0000FF", "Egocentric" = "#008000")

        # If it's an Allocentric plot, assign blue; if it's Egocentric, assign green
        plot_color <- ifelse(plot_type == "Allocentric", "#0000FF", "#008000")

        # Convert the predictor name to a symbol
        predictor_sym <- sym(predictor)
        p <- ggplot() +
            geom_line(data = eff, aes(x = !!predictor_sym, y = fit, color = plot_type), linewidth = 1.2) +
            geom_ribbon(data = eff, aes(x = !!predictor_sym, ymin = lower, ymax = upper, fill = plot_type), alpha = 0.2) +
            scale_color_manual(values = c("Allocentric" = "#0000FF", "Egocentric" = "#008000")) +
            scale_fill_manual(values = c("Allocentric" = "#0000FF", "Egocentric" = "#008000"), guide = 'none') +
            scale_x_continuous(breaks = breaks, limits = xlim_range)  +  
            scale_y_continuous(limits = c(0.55, 1.2), 
                               breaks = c(0.6, 0.7, 0.8, 0.9, 1.0)  
            ) +
            theme_minimal() +
            theme(
              legend.position = if (show_legend) c(0.45, 0.9) else "none",
              legend.title = element_blank(),
              legend.text = element_text(size = 22),
              axis.text.x = element_text(size = 24, color = "black"),
              axis.text.y = element_text(size = 24, color = "black"),
              axis.title.x = element_blank(),
              axis.title.y = element_text(size = 26),
              plot.margin = margin(1, 10, 1, 1),
              panel.grid = element_blank(),
              panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
            ) +
            guides(color = guide_legend(override.aes = list(fill = NA)))

            if (!show_yticks) {
                p <- p + theme(axis.text.y = element_blank(), axis.title.y = element_blank())
            }

            if (!is.null(ylab)) {
                p <- p + labs(y = ylab)
            }

                return(p)
            }
    
    # Extract the effects
    eff <- lmm(cohort2_filtered, predictor)
    eff_allocentric <- eff$eff_allocentric
    eff_egocentric  <- eff$eff_egocentric

    # Allocentric
    plot1 <- create_plot(eff_allocentric, predictor, plot_type = "Allocentric", ylab = "Performance", breaks, show_yticks = TRUE, show_legend = FALSE, xlim_range = xlim_range) +
      ggtitle("Allocentric") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    # Egocentric
    plot2 <- create_plot(eff_egocentric, predictor, plot_type = "Egocentric", ylab = NULL, breaks, show_yticks = FALSE, show_legend = FALSE, xlim_range = xlim_range) +
      ggtitle("Egocentric") +
      theme(plot.title = element_text(size = 26, hjust = 0.5))

    x_label_plot <- ggplot() +
      theme_void() +
      annotate("text", x = 0.5, y = 0.5, label = xlab, size = 9, vjust = 0.4, hjust = 0.5, lineheight = lineheight)

    # Combine the plots
    plots_combined <- plot1 + plot2

    # Add the x-axis label 
    final_plot <- (plots_combined / x_label_plot) + 
      plot_layout(heights = plotheight)  

    # Save the final plot 
    ggsave(
      filename = outputfile, # File name
      plot = final_plot,     # The plot object to save
      width = 6,             # Width in inches
      height = 6,            # Height in inches
      device = "svg"             # Save as svg
    )

    }
    
# Define paths
paths <- list(
  cohort1 = "D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_analysis_cohort1.csv",
  cohort2 = "D:/Publications/GardenGameBehavior/Data/PreProcessing/periods_complete_analysis_cohort2.csv",
  results_cohort1 = "D:/Publications/GardenGameBehavior/Results/Cohort1/",
  results_cohort2 = "D:/Publications/GardenGameBehavior/Results/Cohort2/",
  results_cohort1and2 = "D:/Publications/GardenGameBehavior/Results/Cohorts1and2/",
  figures = "D:/Publications/GardenGameBehavior/Figures/MainFigures/"
)

# Get dataframes
cohort1 <- read.csv(paths$cohort1)
cohort2 <- read.csv(paths$cohort2)

# Remove practice trial for all analyses
cohort1_filtered <- cohort1 %>% filter(TrialIdx > 0)
cohort2_filtered <- cohort2 %>% filter(TrialIdx > 0)

# Combine dataframes
data_complete <- bind_rows(cohort1_filtered, cohort2_filtered)

# TimeSinceEncEnd in minutes
cohort1_filtered$TimeSinceEncEnd_rescaled <- cohort1_filtered$TimeSinceEncEnd / 60000
cohort2_filtered$TimeSinceEncEnd_rescaled <- cohort2_filtered$TimeSinceEncEnd / 60000
data_complete$TimeSinceEncEnd_rescaled <- data_complete$TimeSinceEncEnd / 60000

# Filter for stable and unstable animals
cohort1_stable <- cohort1_filtered %>% filter(StableObj == 'True')
cohort1_unstable <- cohort1_filtered %>% filter(StableObj == 'False')
cohort2_stable <- cohort2_filtered %>% filter(StableObj == 'True')
cohort2_unstable <- cohort2_filtered %>% filter(StableObj == 'False')
data_complete_stable <- data_complete %>% filter(StableObj == 'True')
data_complete_unstable <- data_complete %>% filter(StableObj == 'False')

# Figure 3C
range_trialidx <- range(data_complete$TrialIdx)
breaks_trialidx <- c(0, 20, 40, 60)
outputfile_3C <- file.path(paths$figures, "Figure3C_20250116.svg")
regression_plots_allo_ego("TrialIdx", "Trial index", breaks_trialidx, range_trialidx, outputfile_3C)

# Figure 3D
outputfile_3D <- file.path(paths$figures, "Figure3D_20250116.svg")
regression_plots_stable_unstable("AlloRetRankedPerformance", "TrialIdx", "Trial index", breaks_trialidx, "#0000FF", "#00BFFF", range_trialidx, outputfile_3D)

# Figure 3E
outputfile_3E <- file.path(paths$figures, "Figure3E_20250116.svg")
regression_plots_stable_unstable("EgoRetRankedPerformance", "TrialIdx", "Trial index", breaks_trialidx, "#008000", "#32CD32", range_trialidx, outputfile_3E)

# Figure 5B
range_fence <- range(data_complete$DistObjNearestFence)
breaks_fence <- c(2, 6, 10)
outputfile_5B <- file.path(paths$figures, "Figure5B_20250116.svg")
regression_plots_allo_ego("DistObjNearestFence", "Distance object to nearest boundary (vu)", breaks_fence, range(data_complete$DistObjNearestFence), outputfile_5B)

# Figure 5C
range_corner <- range(data_complete$DistObjNearestCorner)
breaks_corner <- c(5, 10)
outputfile_5C <- file.path(paths$figures, "Figure5C_20250116.svg")
regression_plots_allo_ego("DistObjNearestCorner", "Distance object to nearest corner (vu)", breaks_corner, range_corner, outputfile_5C)

# Figure 5D
range_tree <- range(data_complete$DistObjNearestTree)
breaks_tree <- c(5, 10)
outputfile_5D <- file.path(paths$figures, "Figure5D_20250116.svg")
regression_plots_allo_ego("DistObjNearestTree", "Distance object to nearest landmark (vu)", breaks_tree, range_tree, outputfile_5D)

# Figure 6B
range_player <- range(data_complete$DistObjPlayerStart)
breaks_player <- c(5, 10, 15, 20)
outputfile_6B <- file.path(paths$figures, "Figure6B_20250116.svg")
regression_plots_allo_ego("DistObjPlayerStart", "Distance object to starting position (vu)", breaks_player, range_player, outputfile_6B)

# Figure 7E
range_time <- range(data_complete$TimeSinceEncEnd_rescaled)
breaks_time <- c(0,2)
outputfile_7E <- file.path(paths$figures, "Figure7E_20250116.svg")
regression_plots_7E("TimeSinceEncEnd_rescaled", "Time since end of encoding \n(min)", breaks_time, range_time, outputfile_7E, lineheight = 0.8, plotheight = c(10, 2))

# Figure 8C
range_fence_eye <- range(cohort2_filtered$EyeEncFence)
breaks_fence_eye <- c(0,1)
outputfile_8C <- file.path(paths$figures, "Figure8C_20250116.svg")
labels_fence = list(main = "Fence", subsets = c("North fence", "East fence", "South fence", "West fence"))
allo_colors_fence <- c("North fence" = "#D3D3D3", "East fence" = "#A9A9A9", "South fence" = "#696969", "West fence" = "#708090", "Fence" = "#0000FF")
ego_colors_fence <- c("North fence" = "#D3D3D3", "East fence" = "#A9A9A9", "South fence" = "#696969", "West fence" = "#708090","Fence" = "#008000")
line_types_fence <- c("North fence" = "dashed", "East fence" = "dashed", "South fence" = "dashed", "West fence" = "dashed", "Fence" = "solid")
predictors = c("EyeEncNorthFence", "EyeEncEastFence", "EyeEncSouthFence", "EyeEncWestFence", "EyeEncFence")
regression_plot_8CD(predictors, "Time viewing the boundaries \n(fraction of encoding)", breaks_fence_eye, range_fence_eye, 
                    outputfile_8C, labels = labels_fence, colors_allocentric = allo_colors_fence,
                    colors_egocentric = ego_colors_fence, linetypes = line_types_fence)

# Figure 8D
range_corner_eye <- range(cohort2_filtered$EyeEncCorner)
breaks_corner_eye <- c(0,0.7)
outputfile_8D <- file.path(paths$figures, "Figure8D_20250116.svg")
labels_corner = list(main = "Corner", subsets = c("NE corner", "NW corner", "SW corner", "SE corner"))
allo_colors_corner <- c("NE corner" = "#D3D3D3", "NW corner" = "#A9A9A9", "SW corner" = "#696969", "SE corner" = "#708090", "Corner" = "#0000FF")
ego_colors_corner <- c("NE corner" = "#D3D3D3", "NW corner" = "#A9A9A9", "SW corner" = "#696969", "SE corner" = "#708090","Corner" = "#008000")
line_types_corner <- c("NE corner" = "dashed", "NW corner" = "dashed", "SW corner" = "dashed", "SE corner" = "dashed", "Corner" = "solid")
predictors = c("EyeEncNorthEastCorner", "EyeEncNorthWestCorner", "EyeEncSouthWestCorner", "EyeEncSouthEastCorner", "EyeEncCorner")
regression_plot_8CD(predictors, "Time viewing the corners \n(fraction of encoding)", breaks_corner_eye, range_corner_eye, 
                    outputfile_8D, labels = labels_corner, colors_allocentric = allo_colors_corner,
                    colors_egocentric = ego_colors_corner, linetypes = line_types_corner)

# Figure 8E
range_fence_eye <- range(cohort2_filtered$EyeEncCoverage)
breaks_fence_eye <- c(0,0.3)
outputfile_8E <- file.path(paths$figures, "Figure8E_20250116.svg")
regression_plots_eye("EyeEncCoverage", "Gaze coverage\n", breaks_fence_eye, range_fence_eye, outputfile_8E)

# Figure 8F
range_animal_eye <- range(cohort2_filtered$EyeEncAnimal)
breaks_animal_eye <- c(0,1)
outputfile_8F <- file.path(paths$figures, "Figure8F_20250116.svg")
regression_plots_eye("EyeEncAnimal", "Time viewing the object \n(fraction of encoding)\n", breaks_animal_eye, range_animal_eye, outputfile_8F, plotheight = c(10, 3))

# Figure 8G
range_gazearea_eye <- c(range(cohort2_filtered$EyeEncGazeArea)[1], range(cohort2_filtered$EyeEncGazeArea)[2] + 0.011)
breaks_gazearea_eye <- c(0,0.15)
outputfile_8G <- file.path(paths$figures, "Figure8G_20250116.svg")
regression_plots_eye("EyeEncGazeArea", "Time viewing gaze area \n(fraction of encoding)\n", breaks_gazearea_eye, range_gazearea_eye, outputfile_8G, plotheight = c(10, 3))

# Figure 8H
range_animal_gazearea_eye <- range(cohort2_filtered$EyeEncGazeAreaAndAnimal)
breaks_animal_gazearea_eye <- c(0,1)
outputfile_8H <- file.path(paths$figures, "Figure8H_20250116.svg")
regression_plots_eye("EyeEncGazeAreaAndAnimal", "Time viewing the \n object and gaze area \n(fraction of encoding)", breaks_animal_gazearea_eye, range_animal_gazearea_eye, outputfile_8H, plotheight = c(10, 3))