# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

delegated: static HTML/CSS. This repository is the web surface for a local CLI; a static site can explain and convert without implying a hosted training app. Framework choice is open if a later surface needs one.

## Users

People who want to train models on a machine they already have. They install Troy locally and need a training run without assembling a training stack themselves.

Exact persona (researcher, ML engineer, hobbyist, or other) is an open decision.

## Product Purpose

Troy is a CLI installed on the user's machine. It is an abstraction for training models locally.

Success is that a user can train a model on their machine by giving Troy a YAML config file, rather than by building a training pipeline.

## Positioning

Training is as easy as writing a YAML config and handing it to Troy. The product is the local abstraction and that config contract, not a notebook, a cloud trainer, or a dashboard that runs training elsewhere.

What kinds of models Troy trains is an open decision.

## Operating Context

- Troy runs as a CLI on the user's own machine.
- A YAML config file is the way a training run is expressed.
- This repository (`web`) is the web surface for that CLI product. Which web surface comes first (marketing, docs, or another) is an open decision.

## Capabilities and Constraints

- Local model training via a YAML config abstraction.
- Installed on the user's machine; training happens there.
- Greenfield: no existing implementation, brand system, or proof assets in this repository.
- Undecided: model types, exact primary persona, and the first web surface to build.

## Brand Commitments

The product name is Troy. No other name, voice, logo, or identity constraint was made binding.

Standing visual preference (init direction round, seed 23ef818a): the category standard, played straight. Craft bar is Cursor, Linear, and Vercel — their finish, not their products. No irony, no smuggled quirk.

## Evidence on Hand

None. No customers, benchmarks, testimonials, screenshots, or training-run artifacts were provided. Future work must not fabricate them.

## Product Principles

1. **The machine is the venue.** Training runs on the user's machine, not on infrastructure Troy operates.
2. **The YAML file is the interface.** A training run is specified in config, not in a hidden wizard or a pile of scripts.
3. **Ease is the product.** Troy exists to make local training easy; completeness of every training knob is not the claim.
4. **Do not pretend to be a hosted ML platform.** Cloud notebooks, remote clusters, and dashboards that train elsewhere are neighboring products, not this one.
5. **Do not invent proof.** No customers, numbers, or case studies exist yet; do not write them into the product.
