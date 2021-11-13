---
layout: page
permalink: /research/
title: Research
order: 3
---



{% for post in site.categories.research %}
 <li><span>{{ post.date | date_to_string }}</span> &nbsp; <a href="{{ post.url }}">{{ post.title }}</a></li>
{% endfor %}