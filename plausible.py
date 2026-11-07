#!/usr/bin/env python3
"""
Commonsense filter for (subject class, predicate, object class).

Drops relations no reader would accept, such as "door walking on window" or
"sky talking to car". The rules follow from what each predicate means and were
written from the class and predicate lists alone, not fitted to GT. The tubes keep
their SAM3 class, so a mislabelled tube is judged by its wrong label. Class
'others' is unknown and never judged.
"""
ANIMATE = {'person', 'bird', 'dog'}
VEHICLE = {'car', 'motorcycle', 'truck'}
STUFF = {'sky', 'floor', 'ceiling', 'ground', 'wall', 'grass', 'fence', 'stair', 'window'}
FIXED = STUFF | {'door', 'road_barrier'}
SMALL = {'handbag', 'bottle', 'cup', 'bowl', 'cellphone', 'stick', 'laptop'}
OPENABLE = {'door', 'window', 'car', 'truck', 'box', 'bag', 'handbag', 'suitcase', 'bin',
            'laptop', 'bottle'}

# subject requirements
AGENT = {'caressing', 'catching', 'closing', 'dancing on', 'drinking with', 'driving',
         'fighting with', 'holding', 'jumping from', 'jumping on', 'kneeling to', 'lifting up',
         'looking at', 'opening', 'picking', 'playing', 'playing with', 'pointing to', 'riding',
         'robbing', 'running on', 'stepping on', 'taking out', 'talking to', 'throwing',
         'walking on', 'walking to', 'watching'}
AGENT_OR_VEHICLE = {'carrying', 'entering', 'following', 'hitting', 'pulling', 'pushing'}
AT_REST = {'sitting on', 'standing on', 'lying on', 'hanging from'}   # also "bottle sitting on table"
NOT_STUFF_SUBJECT = {'moving', 'touching', 'in', 'on', 'next to'}

# object requirements
SOCIAL = {'caressing', 'fighting with', 'kneeling to', 'robbing', 'talking to'}
SUPPORT = {'dancing on', 'jumping on', 'lying on', 'running on', 'sitting on', 'standing on',
           'stepping on', 'walking on'}
NOT_SUPPORT = {'sky', 'ceiling', 'wall', 'window', 'door'} | SMALL
LOCOMOTION = {'dancing on', 'running on', 'walking on'}
GRASP = {'carrying', 'catching', 'holding', 'lifting up', 'picking', 'taking out', 'throwing'}


def plausible(s, p, o):
    """-> None if acceptable, else the rule that rejected it."""
    if s == 'others' or o == 'others' or s is None or o is None:
        return None
    if p in AGENT and s not in ANIMATE:
        return 'subject must be animate'
    if p in AGENT_OR_VEHICLE and s not in ANIMATE | VEHICLE:
        return 'subject must be animate or a vehicle'
    if p in AT_REST and s in FIXED:
        return 'subject cannot rest on something'
    if p in NOT_STUFF_SUBJECT and s in STUFF:
        return 'scenery cannot be the subject'
    if p in SOCIAL and o not in ANIMATE:
        return 'object must be animate'
    if p in SUPPORT and o in NOT_SUPPORT:
        return 'object cannot support the subject'
    if p in LOCOMOTION and o in ANIMATE:
        return 'cannot walk on an animate object'
    if p in GRASP and o in FIXED | VEHICLE and not (p == 'holding' and o == 'door'):
        return 'object cannot be picked up'     # "holding a door" open is fine
    if p in ('driving', 'riding') and o not in VEHICLE:
        return 'object must be a vehicle'
    if p == 'entering' and o not in VEHICLE | {'door', 'window'}:
        return 'object cannot be entered'
    if p in ('opening', 'closing') and o not in OPENABLE:
        return 'object cannot be opened'
    if p == 'drinking with' and o not in {'bottle', 'cup', 'bowl', 'person'}:
        return 'object is not a drink or a person'
    if p == 'in' and o in {'sky', 'wall', 'ceiling', 'window', 'door'}:
        return 'object is not a container'
    if p == 'on' and o in {'sky', 'window'}:
        return 'object is not a surface'
    if p == 'following' and o not in ANIMATE | VEHICLE:
        return 'object cannot be followed'
    if p in ('pushing', 'pulling', 'playing with', 'playing') and o in STUFF:
        return 'object is scenery'
    if p in ('walking to', 'jumping from', 'hitting') and o in {'sky', 'ceiling'}:
        return 'object is out of reach'
    return None
