use std::{collections::HashMap, env::var};

use itertools::Itertools;

use crate::{
    air::{
        air::{AirExpression, Bus, Direction, Interaction},
        constructor::{AirConstructor, TimestampUsage},
    },
    folder1::{
        file2_tree::{ExpressionContainer, ScopePath},
        file3::{FlatFunctionCall, FlatMatch, FlatStatement},
        function_resolution::Stage,
        ir::{ArithmeticOperator, Expression, Material, Statement, Type},
        type_resolution::TypeSet,
    },
};

#[derive(Default)]
pub struct Representation {
    pub expressions: Vec<AirExpression>,
    pub owned: Vec<bool>,
}

#[derive(Default)]
pub struct RepresentationTable {
    representations: HashMap<(ScopePath, String), Representation>,
}

impl RepresentationTable {
    fn get_representation(&mut self, scope: &ScopePath, name: &String) -> Vec<AirExpression> {
        for prefix in scope.prefixes() {
            let representation = self.representations.get(&(prefix.clone(), name.clone()));
            if let Some(representation) = representation {
                return representation.expressions.clone();
            }
        }
        unreachable!()
    }
    pub fn fill_in_and_add_representation(
        &mut self,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
        name: &String,
        right: &mut [Option<AirExpression>],
    ) {
        let mut representation = Representation::default();
        for right in right.iter_mut() {
            if let Some(right) = right {
                representation.expressions.push(right.clone());
                representation.owned.push(false);
            } else {
                let new_cell = air_constructor.new_normal_cell(scope);
                let expression = AirExpression::single_cell(new_cell);
                representation.expressions.push(expression.clone());
                representation.owned.push(true);
                *right = Some(expression);
            }
        }
        self.representations
            .insert((scope.clone(), name.clone()), representation);
    }
    pub fn add_representation(
        &mut self,
        scope: &ScopePath,
        name: &String,
        representation: Vec<AirExpression>,
    ) {
        let representation_len = representation.len();
        self.representations.insert(
            (scope.clone(), name.clone()),
            Representation {
                expressions: representation,
                owned: vec![false; representation_len],
            },
        );
    }
}

impl ExpressionContainer {
    pub fn calc_representation(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        scope: &ScopePath,
    ) -> Vec<AirExpression> {
        match self.expression.as_ref() {
            Expression::Constant { value } => {
                vec![AirExpression::constant(*value)]
            }
            Expression::Variable {
                name, represents, ..
            } => {
                assert!(!*represents);
                representation_table.get_representation(scope, name)
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let type_name = &type_set.constructors[constructor].0;
                let tipo = &type_set.algebraic_types[type_name];
                let mut result = Vec::new();
                if tipo.variants.len() != 1 {
                    let i = tipo
                        .variants
                        .iter()
                        .position(|variant| &variant.name == constructor)
                        .unwrap();
                    result.push(AirExpression::constant(i as isize));
                }
                for field in fields {
                    result.extend(field.calc_representation(type_set, representation_table, scope));
                }
                let type_length = type_set.calc_type_size(&Type::NamedType(type_name.clone()));
                while result.len() < type_length {
                    result.push(AirExpression::zero());
                }
                result
            }
            Expression::Arithmetic {
                operator,
                left,
                right,
            } => {
                let left = &left.calc_representation(type_set, representation_table, scope)[0];
                let right = &right.calc_representation(type_set, representation_table, scope)[0];
                vec![match operator {
                    ArithmeticOperator::Plus => left.plus(right),
                    ArithmeticOperator::Minus => left.minus(right),
                    ArithmeticOperator::Times => left.times(right),
                    ArithmeticOperator::Div => unreachable!(),
                }]
            }
            Expression::Dematerialized { .. } => vec![],
            Expression::Eq { .. } => unreachable!(),
            Expression::EmptyConstArray { .. } => vec![],
            Expression::ConstArray { elements } => elements
                .iter()
                .flat_map(|element| {
                    element.calc_representation(type_set, representation_table, scope)
                })
                .collect(),
            Expression::ConstArrayConcatenation { left, right } => left
                .calc_representation(type_set, representation_table, scope)
                .into_iter()
                .chain(right.calc_representation(type_set, representation_table, scope))
                .collect(),
            Expression::ConstArrayAccess { array, index } => {
                let array_representation =
                    array.calc_representation(type_set, representation_table, scope);
                let elem_size = type_set.calc_type_size(self.get_type());
                array_representation[index * elem_size..(index + 1) * elem_size].to_vec()
            }
            Expression::ConstArraySlice { array, from, to } => {
                let array_representation =
                    array.calc_representation(type_set, representation_table, scope);
                let (elem_type, _) = array
                    .get_type()
                    .get_const_array_type(Material::Materialized)
                    .unwrap();
                let elem_size = type_set.calc_type_size(elem_type);
                array_representation[from * elem_size..to * elem_size].to_vec()
            }
            Expression::ConstArrayRepeated { element, length } => {
                let element_representation =
                    element.calc_representation(type_set, representation_table, scope);
                let elem_type = element.get_type();
                let elem_size = type_set.calc_type_size(elem_type);
                let mut result = Vec::with_capacity(*length * elem_size);
                for _ in 0..*length {
                    result.extend(element_representation.clone());
                }
                result
            }
        }
    }

    pub fn represent_top_down(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
        representation: &mut [Option<AirExpression>],
    ) {
        match self.expression.as_ref() {
            Expression::Constant { value } => {
                air_constructor.add_scoped_constraint(
                    scope,
                    vec![AirExpression::constant(*value)],
                    representation,
                );
            }
            Expression::Variable {
                name,
                represents: true,
                ..
            } => {
                representation_table.fill_in_and_add_representation(
                    air_constructor,
                    scope,
                    name,
                    representation,
                );
            }
            Expression::Algebraic {
                constructor,
                fields,
            } => {
                let type_name = &type_set.constructors[constructor].0;
                let tipo = &type_set.algebraic_types[type_name];
                let mut offset = 0;
                if tipo.variants.len() != 1 {
                    let i = tipo
                        .variants
                        .iter()
                        .position(|variant| &variant.name == constructor)
                        .unwrap();
                    air_constructor.add_scoped_constraint(
                        scope,
                        vec![AirExpression::constant(i as isize)],
                        &mut representation[0..1],
                    );
                    offset += 1;
                }
                for field in fields {
                    let type_length = type_set.calc_type_size(field.get_type());
                    field.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        scope,
                        &mut representation[offset..offset + type_length],
                    );
                    offset += type_length;
                }
            }
            Expression::Dematerialized { .. } => {}
            Expression::ConstArray { elements } => {
                let (elem_type, _) = self
                    .get_type()
                    .get_const_array_type(Material::Materialized)
                    .unwrap();
                let elem_size = type_set.calc_type_size(elem_type);
                for (i, element) in elements.iter().enumerate() {
                    element.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        scope,
                        &mut representation[i * elem_size..(i + 1) * elem_size],
                    );
                }
            }
            _ => {
                let defined_representation =
                    self.calc_representation(type_set, representation_table, scope);
                air_constructor.add_scoped_constraint(
                    scope,
                    defined_representation,
                    representation,
                );
            }
        }
    }

    pub fn create_representation_top_down(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
    ) -> Vec<AirExpression> {
        let mut representation = vec![None; type_set.calc_type_size(self.get_type())];
        self.represent_top_down(
            type_set,
            representation_table,
            air_constructor,
            scope,
            &mut representation,
        );
        representation.into_iter().map(Option::unwrap).collect()
    }

    pub fn represent_top_down_fixed(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
        scope: &ScopePath,
        representation: &Vec<AirExpression>,
    ) {
        let mut representation: Vec<_> = representation.iter().cloned().map(Some).collect();
        self.represent_top_down(
            type_set,
            representation_table,
            air_constructor,
            scope,
            &mut representation,
        );
    }
}

impl FlatMatch {
    pub fn enforce(
        &self,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
    ) {
        if self.material == Material::Materialized {
            let mut representation = vec![None; type_set.calc_type_size(self.value.get_type())];

            let type_name = self
                .value
                .get_type()
                .get_named_type(Material::Materialized)
                .unwrap();
            let tipo = &type_set.algebraic_types[type_name];
            if tipo.variants.len() != 1 {
                let mut variant_expression = AirExpression::zero();
                for (constructor, _) in self.branches.iter() {
                    let scope = self.scope.then(self.index, constructor.clone());
                    let variant_index = tipo
                        .variants
                        .iter()
                        .position(|variant| &variant.name == constructor)
                        .unwrap();
                    variant_expression = variant_expression.plus(
                        &air_constructor
                            .get_scope_expression(&scope)
                            .times(&AirExpression::constant(variant_index as isize)),
                    );
                }
                representation[0] = Some(variant_expression);
            }

            self.value.represent_top_down(
                type_set,
                representation_table,
                air_constructor,
                &self.scope,
                &mut representation,
            );

            let representation: Vec<_> = representation.into_iter().map(|x| x.unwrap()).collect();

            for (constructor, components) in self.branches.iter() {
                let mut offset = if tipo.variants.len() == 1 { 0 } else { 1 };
                let scope = self.scope.then(self.index, constructor.clone());
                for (component, tipo) in components.iter().zip_eq(
                    tipo.variants
                        .iter()
                        .find(|variant| &variant.name == constructor)
                        .unwrap()
                        .components
                        .iter(),
                ) {
                    let type_size = type_set.calc_type_size(tipo);
                    representation_table.add_representation(
                        &scope,
                        component,
                        representation[offset..offset + type_size].to_vec(),
                    );
                    offset += type_size;
                }
            }
        }
    }
}

impl FlatFunctionCall {
    pub fn represent_stage(
        &self,
        interaction: &mut Interaction,
        stage: Stage,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
    ) {
        for i in stage.start..stage.mid {
            let argument = &self.arguments[i];
            interaction.fields.extend(argument.calc_representation(
                type_set,
                representation_table,
                &ScopePath::empty(),
            ));
        }
        for i in stage.mid..stage.end {
            interaction
                .fields
                .extend(self.arguments[i].create_representation_top_down(
                    type_set,
                    representation_table,
                    air_constructor,
                    &ScopePath::empty(),
                ));
        }
    }
}

impl FlatStatement {
    pub fn enforce(
        &self,
        index: usize,
        type_set: &TypeSet,
        representation_table: &mut RepresentationTable,
        air_constructor: &mut AirConstructor,
    ) {
        if self.material == Material::Materialized {
            match &self.statement {
                Statement::VariableDeclaration {
                    name,
                    tipo,
                    represents,
                } => {
                    if *represents {
                        representation_table.fill_in_and_add_representation(
                            air_constructor,
                            &self.scope,
                            name,
                            &mut vec![None; type_set.calc_type_size(tipo)],
                        );
                    }
                }
                Statement::Equality { left, right } => {
                    let representation =
                        right.calc_representation(type_set, representation_table, &self.scope);
                    let mut representation: Vec<_> = representation.into_iter().map(Some).collect();
                    left.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &mut representation,
                    );
                }
                Statement::Reference { reference, data } => {
                    let data_representation =
                        data.calc_representation(type_set, representation_table, &self.scope);
                    let reference_representation = reference.create_representation_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                    );
                    let multiplicity_cell =
                        air_constructor.new_multiplicity_cell(index, &self.scope);
                    let mut fields = vec![];
                    fields.extend(reference_representation);
                    fields.extend(data_representation);
                    air_constructor.add_scoped_interaction(
                        &self.scope,
                        Interaction {
                            bus: Bus::Memory,
                            direction: Direction::Receive,
                            multiplicity: AirExpression::single_cell(multiplicity_cell),
                            fields,
                        },
                    );
                }
                Statement::Dereference { data, reference } => {
                    let reference_representation =
                        reference.calc_representation(type_set, representation_table, &self.scope);
                    let data_representation = data.create_representation_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                    );
                    let mut fields = vec![];
                    fields.extend(reference_representation);
                    fields.extend(data_representation);
                    air_constructor.add_scoped_interaction(
                        &self.scope,
                        Interaction {
                            bus: Bus::Memory,
                            direction: Direction::Send,
                            multiplicity: AirExpression::one(),
                            fields,
                        },
                    );
                }
                Statement::EmptyUnderConstructionArray { array, elem_type } => {
                    let mut representation = vec![None, Some(AirExpression::zero())];
                    array.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &mut representation,
                    );
                }
                Statement::UnderConstructionArrayPrepend {
                    new_array,
                    elem,
                    old_array,
                } => {
                    let old_array_representation =
                        old_array.calc_representation(type_set, representation_table, &self.scope);
                    let elem_representation =
                        elem.calc_representation(type_set, representation_table, &self.scope);
                    let mut new_array_representation = vec![
                        Some(old_array_representation[0].minus(&AirExpression::one())),
                        Some(old_array_representation[1].plus(&AirExpression::one())),
                    ];
                    new_array.represent_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &mut new_array_representation,
                    );

                    let multiplicity_cell =
                        air_constructor.new_multiplicity_cell(index, &self.scope);
                    let mut fields = vec![];
                    fields.push(new_array_representation[0].as_ref().unwrap().clone());
                    fields.extend(elem_representation);
                    air_constructor.add_scoped_interaction(
                        &self.scope,
                        Interaction {
                            bus: Bus::Memory,
                            direction: Direction::Receive,
                            multiplicity: AirExpression::single_cell(multiplicity_cell),
                            fields,
                        },
                    );
                }
                Statement::ArrayFinalization {
                    finalized,
                    under_construction,
                } => {
                    let under_construction_representation = under_construction.calc_representation(
                        type_set,
                        representation_table,
                        &self.scope,
                    );
                    finalized.represent_top_down_fixed(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                        &under_construction_representation,
                    );
                    let array_length = &under_construction_representation[1];
                    let (before, after) = &air_constructor.timestamp_before_after
                        [&TimestampUsage::FinalizeArray(index)];
                    air_constructor.add_scoped_single_constraint(
                        &self.scope,
                        before.plus(array_length),
                        AirExpression::single_cell(*after),
                    );
                }
                Statement::ArrayAccess { array, index, elem } => {
                    let array_representation =
                        array.calc_representation(type_set, representation_table, &self.scope);
                    let index_representation =
                        index.calc_representation(type_set, representation_table, &self.scope);
                    let elem_representation = elem.create_representation_top_down(
                        type_set,
                        representation_table,
                        air_constructor,
                        &self.scope,
                    );
                    let pointer = array_representation[0].plus(&index_representation[0]);
                    let mut fields = vec![pointer];
                    fields.extend(elem_representation);
                    air_constructor.add_scoped_interaction(
                        &self.scope,
                        Interaction {
                            bus: Bus::Memory,
                            direction: Direction::Send,
                            multiplicity: AirExpression::one(),
                            fields,
                        },
                    );
                }
            }
        }
    }
}
