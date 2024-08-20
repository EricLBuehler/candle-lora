use proc_macro::TokenStream as TokenStream1;
use proc_macro2::TokenStream;
use syn::{
    parse::Parser, parse_macro_input, Data, DeriveInput, Fields, GenericArgument, Ident,
    PathArguments, Type, TypeParamBound, Visibility,
};

#[proc_macro_attribute]
pub fn replace_layer_fields(_args: TokenStream1, input: TokenStream1) -> TokenStream1 {
    let mut ast = parse_macro_input!(input as DeriveInput);
    match &mut ast.data {
        Data::Struct(ref mut struct_data) => {
            match &mut struct_data.fields {
                Fields::Named(fields) => {
                    for field in fields.named.iter_mut() {
                        let mut f = None;
                        let ident = field.ident.clone().unwrap();
                        let ty = field.ty.clone();
                        if let Type::Path(path) = ty {
                            if path.path.segments.len() == 1 {
                                match path
                                    .path
                                    .segments
                                    .first()
                                    .unwrap()
                                    .ident
                                    .to_string()
                                    .as_str()
                                {
                                    "Linear" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Arc<dyn LinearLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Arc<dyn LinearLayerLike>)).unwrap());
                                        }
                                    }
                                    "Conv1d" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Arc<dyn Conv1dLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Arc<dyn Conv1dLayerLike>)).unwrap());
                                        }
                                    }
                                    "Conv2d" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Arc<dyn Conv2dLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Arc<dyn Conv2dLayerLike>)).unwrap());
                                        }
                                    }
                                    "Embedding" => {
                                        if let Visibility::Public(_) = field.vis {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Arc<dyn EmbeddingLayerLike>)).unwrap());
                                        } else {
                                            f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Arc<dyn EmbeddingLayerLike>)).unwrap());
                                        }
                                    }
                                    "Option" => {
                                        if let PathArguments::AngleBracketed(bracketed) =
                                            &path.path.segments.first().unwrap().arguments
                                        {
                                            if bracketed.args.len() == 1 {
                                                if let GenericArgument::Type(Type::Path(tp)) =
                                                    bracketed.args.first().unwrap()
                                                {
                                                    if tp.path.segments.len() == 1 {
                                                        match tp
                                                            .path
                                                            .segments
                                                            .first()
                                                            .unwrap()
                                                            .ident
                                                            .to_string()
                                                            .as_str()
                                                        {
                                                            "Linear" => {
                                                                if let Visibility::Public(_) =
                                                                    field.vis
                                                                {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Option<Arc<dyn LinearLayerLike>>)).unwrap());
                                                                } else {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Option<Arc<dyn LinearLayerLike>>)).unwrap());
                                                                }
                                                            }
                                                            "Conv1d" => {
                                                                if let Visibility::Public(_) =
                                                                    field.vis
                                                                {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Option<Arc<dyn Conv1dLayerLike>>)).unwrap());
                                                                } else {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Option<Arc<dyn Conv1dLayerLike>>)).unwrap());
                                                                }
                                                            }
                                                            "Conv2d" => {
                                                                if let Visibility::Public(_) =
                                                                    field.vis
                                                                {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Option<Arc<dyn Conv2dLayerLike>>)).unwrap());
                                                                } else {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Option<Arc<dyn Conv2dLayerLike>>)).unwrap());
                                                                }
                                                            }
                                                            "Embedding" => {
                                                                if let Visibility::Public(_) =
                                                                    field.vis
                                                                {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(pub #ident: Option<Arc<dyn EmbeddingLayerLike>>)).unwrap());
                                                                } else {
                                                                    f = Some(syn::Field::parse_named.parse2(quote::quote!(#ident: Option<Arc<dyn EmbeddingLayerLike>>)).unwrap());
                                                                }
                                                            }
                                                            _ => {}
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if let Some(f) = f {
                            *field = f;
                        }
                    }
                }
                _ => {
                    panic!("Named fields are required.")
                }
            }
        }
        _ => {
            panic!("Cannot swap fields of non struct!");
        }
    }

    quote::quote!(#ast).into()
}

fn is_ident(ident: &Ident, name: &str) -> bool {
    *ident == name
}

#[proc_macro_derive(AutoLoraConvert)]
pub fn auto_lora_convert(tokens: TokenStream1) -> TokenStream1 {
    let ast = parse_macro_input!(tokens as DeriveInput);
    let mut linear_fields = Vec::new();
    let mut conv1d_fields = Vec::new();
    let mut conv2d_fields = Vec::new();
    let mut embed_fields = Vec::new();

    let mut linear_option1_fields = Vec::new();
    let mut conv1d_option1_fields = Vec::new();
    let mut conv2d_option1_fields = Vec::new();
    let mut embed_option1_fields = Vec::new();
    let st_name = &ast.ident;

    match ast.data {
        Data::Struct(st) => {
            for field in st.fields {
                match field.ty {
                    Type::Path(path) => {
                        let segments = path.path.segments.into_iter().collect::<Vec<_>>();
                        if segments.len() != 1 {
                            continue;
                        }
                        if is_ident(&segments[0].ident, "Option") {
                            if let syn::PathArguments::AngleBracketed(bracketed) =
                                &segments.first().as_ref().unwrap().arguments
                            {
                                if bracketed.args.len() != 1 {
                                    continue;
                                }
                                if let GenericArgument::Type(Type::Path(ty)) = &bracketed.args[0] {
                                    if ty.path.segments.len() != 1 {
                                        continue;
                                    }
                                    let typname = &ty.path.segments.first().unwrap().ident;
                                    if is_ident(typname, "Arc") {
                                        if let syn::PathArguments::AngleBracketed(bracketed) =
                                            &ty.path.segments.first().as_ref().unwrap().arguments
                                        {
                                            if bracketed.args.len() != 1 {
                                                continue;
                                            }
                                            match &bracketed.args[0] {
                                                GenericArgument::Type(Type::TraitObject(trobj)) => {
                                                    let bounds = &trobj.bounds;
                                                    if bounds.len() != 1 {
                                                        continue;
                                                    }
                                                    match bounds.first().unwrap() {
                                                        TypeParamBound::Trait(bound) => {
                                                            if bound.path.segments.len() != 1 {
                                                                continue;
                                                            }
                                                            let trt = &bound
                                                                .path
                                                                .segments
                                                                .first()
                                                                .unwrap()
                                                                .ident;
                                                            let value = (
                                                                field.ident.clone().unwrap(),
                                                                field
                                                                    .ident
                                                                    .as_ref()
                                                                    .unwrap()
                                                                    .to_string(),
                                                            );
                                                            if is_ident(trt, "LinearLayerLike") {
                                                                linear_option1_fields.push(value);
                                                            } else if is_ident(
                                                                trt,
                                                                "Conv1dLayerLike",
                                                            ) {
                                                                conv1d_option1_fields.push(value);
                                                            } else if is_ident(
                                                                trt,
                                                                "Conv2dLayerLike",
                                                            ) {
                                                                conv2d_option1_fields.push(value);
                                                            } else if is_ident(
                                                                trt,
                                                                "EmbeddingLayerLike",
                                                            ) {
                                                                embed_option1_fields.push(value);
                                                            }
                                                        }
                                                        _ => continue,
                                                    }
                                                }
                                                _ => continue,
                                            }
                                        } else {
                                            continue;
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        if !is_ident(&segments[0].ident, "Arc") {
                            continue;
                        }
                        if let syn::PathArguments::AngleBracketed(bracketed) =
                            &segments.first().as_ref().unwrap().arguments
                        {
                            if bracketed.args.len() != 1 {
                                continue;
                            }
                            match &bracketed.args[0] {
                                GenericArgument::Type(Type::TraitObject(trobj)) => {
                                    let bounds = &trobj.bounds;
                                    if bounds.len() != 1 {
                                        continue;
                                    }
                                    match bounds.first().unwrap() {
                                        TypeParamBound::Trait(bound) => {
                                            if bound.path.segments.len() != 1 {
                                                continue;
                                            }
                                            let trt = &bound.path.segments.first().unwrap().ident;
                                            let value = (
                                                field.ident.clone().unwrap(),
                                                field.ident.as_ref().unwrap().to_string(),
                                            );
                                            if is_ident(trt, "LinearLayerLike") {
                                                linear_fields.push(value);
                                            } else if is_ident(trt, "Conv1dLayerLike") {
                                                conv1d_fields.push(value);
                                            } else if is_ident(trt, "Conv2dLayerLike") {
                                                conv2d_fields.push(value);
                                            } else if is_ident(trt, "EmbeddingLayerLike") {
                                                embed_fields.push(value);
                                            }
                                        }
                                        _ => continue,
                                    }
                                }
                                _ => continue,
                            }
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                }
            }
        }
        _ => {
            todo!()
        }
    }

    let mut linear_stream = TokenStream::new();
    if !linear_fields.is_empty() {
        quote_into::quote_into!(linear_stream += [#{
            for (namei,name) in linear_fields.iter() {
                quote_into::quote_into!(linear_stream += (linear.insert(#name.to_string(), &*self.#namei)),)
            }
        }];);
    }

    let mut linear_get = TokenStream::new();
    if !linear_fields.is_empty() {
        quote_into::quote_into!(linear_get += [#{
            for (namei,_) in linear_fields.iter() {
                quote_into::quote_into!(linear_get += (self.#namei.get_tensors(&mut output)),)
            }
        }];);
    }

    let mut conv1d_stream = TokenStream::new();
    if !conv1d_fields.is_empty() {
        quote_into::quote_into!(conv1d_stream += [#{
            for (namei,name) in conv1d_fields.iter() {
                quote_into::quote_into!(conv1d_stream += (conv1d.insert(#name.to_string(), &*self.#namei)),)
            }
        }];);
    }

    let mut conv1d_get = TokenStream::new();
    if !conv1d_fields.is_empty() {
        quote_into::quote_into!(conv1d_get += [#{
            for (namei,_) in conv1d_fields.iter() {
                quote_into::quote_into!(conv1d_get += (self.#namei.get_tensors(&mut output)),)
            }
        }];);
    }

    let mut conv2d_stream = TokenStream::new();
    if !conv2d_fields.is_empty() {
        quote_into::quote_into!(conv2d_stream += [#{
            for (namei,name) in conv2d_fields.iter() {
                quote_into::quote_into!(conv2d_stream += (conv2d.insert(#name.to_string(), &*self.#namei)),)
            }
        }];);
    }

    let mut conv2d_get = TokenStream::new();
    if !conv2d_fields.is_empty() {
        quote_into::quote_into!(conv2d_get += [#{
            for (namei,_) in conv2d_fields.iter() {
                quote_into::quote_into!(conv2d_get += (self.#namei.get_tensors(&mut output)),)
            }
        }];);
    }

    let mut embed_stream = TokenStream::new();
    if !embed_fields.is_empty() {
        quote_into::quote_into!(embed_stream += [#{
            for (namei,name) in embed_fields.iter() {
                quote_into::quote_into!(embed_stream += (embed.insert(#name.to_string(), &*self.#namei)),)
            }
        }];);
    }

    let mut embed_get = TokenStream::new();
    if !embed_fields.is_empty() {
        quote_into::quote_into!(embed_get += [#{
            for (namei,_) in embed_fields.iter() {
                quote_into::quote_into!(embed_get += (self.#namei.get_tensors(&mut output)),)
            }
        }];);
    }

    let mut linear_stream_assign = TokenStream::new();
    if !linear_fields.is_empty() {
        quote_into::quote_into!(linear_stream_assign += [#{
            for (name, n) in linear_fields.iter() {
                linear_stream_assign.extend(quote::quote!((self.#name = ::std::sync::Arc::new(new_layers.linear.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut linear_merge_stream_assign = TokenStream::new();
    if !linear_fields.is_empty() {
        quote_into::quote_into!(linear_merge_stream_assign += [#{
            for (name, n) in linear_fields.iter() {
                linear_merge_stream_assign.extend(quote::quote!(({
                    (new_layers.linear.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for linear.");
                    self.#name = ::std::sync::Arc::new(new_layers.linear.get(#n).unwrap().clone())
                }),))
            }
        }];);
    }

    let mut conv1d_stream_assign = TokenStream::new();
    if !conv1d_fields.is_empty() {
        quote_into::quote_into!(conv1d_stream_assign += [#{
            for (name, n) in conv1d_fields.iter() {
                conv1d_stream_assign.extend(quote::quote!((self.#name = ::std::sync::Arc::new(new_layers.conv1d.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut conv1d_merge_stream_assign = TokenStream::new();
    if !conv1d_fields.is_empty() {
        quote_into::quote_into!(conv1d_merge_stream_assign += [#{
            for (name, n) in conv1d_fields.iter() {
                conv1d_merge_stream_assign.extend(quote::quote!(({
                    (new_layers.conv1d.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for conv1d.");
                    self.#name = ::std::sync::Arc::new(new_layers.conv1d.get(#n).unwrap().clone())
                }),))
            }
        }];);
    }

    let mut conv2d_stream_assign = TokenStream::new();
    if !conv2d_fields.is_empty() {
        quote_into::quote_into!(conv2d_stream_assign += [#{
            for (name, n) in conv2d_fields.iter() {
                conv2d_stream_assign.extend(quote::quote!((self.#name = ::std::sync::Arc::new(new_layers.conv2d.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut conv2d_merge_stream_assign = TokenStream::new();
    if !conv2d_fields.is_empty() {
        quote_into::quote_into!(conv2d_merge_stream_assign += [#{
            for (name, n) in conv2d_fields.iter() {
                conv2d_merge_stream_assign.extend(quote::quote!(({
                    (new_layers.conv2d.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for conv2d.");
                    self.#name = ::std::sync::Arc::new(new_layers.conv2d.get(#n).unwrap().clone())
                }),))
            }
        }];);
    }

    let mut embed_stream_assign = TokenStream::new();
    if !embed_fields.is_empty() {
        quote_into::quote_into!(embed_stream_assign += [#{
            for (name, n) in embed_fields.iter() {
                embed_stream_assign.extend(quote::quote!((self.#name = ::std::sync::Arc::new(new_layers.embed.get(#n).unwrap().clone())),))
            }
        }];);
    }

    let mut embed_merge_stream_assign = TokenStream::new();
    if !embed_fields.is_empty() {
        quote_into::quote_into!(embed_merge_stream_assign += [#{
            for (name, n) in embed_fields.iter() {
                embed_merge_stream_assign.extend(quote::quote!(({
                    (new_layers.embed.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for embed.");
                    self.#name = ::std::sync::Arc::new(new_layers.embed.get(#n).unwrap().clone())
                }),))
            }
        }];);
    }

    let mut linear_option1_stream = TokenStream::new();
    if !linear_option1_fields.is_empty() {
        quote_into::quote_into!(linear_option1_stream += [#{
            for (namei,name) in linear_option1_fields.iter() {
                quote_into::quote_into!(linear_option1_stream += (linear.insert(#name.to_string(), self.#namei.as_deref().unwrap())),)
            }
        }];);
    }

    let mut conv1d_option1_stream = TokenStream::new();
    if !conv1d_option1_fields.is_empty() {
        quote_into::quote_into!(conv1d_option1_stream += [#{
            for (namei,name) in conv1d_option1_fields.iter() {
                quote_into::quote_into!(conv1d_option1_stream += (conv1d.insert(#name.to_string(), self.#namei.as_deref().unwrap())),)
            }
        }];);
    }

    let mut conv2d_option1_stream = TokenStream::new();
    if !conv2d_option1_fields.is_empty() {
        quote_into::quote_into!(conv2d_option1_stream += [#{
            for (namei,name) in conv2d_option1_fields.iter() {
                quote_into::quote_into!(conv2d_option1_stream += (conv2d.insert(#name.to_string(), self.#namei.as_deref().unwrap())),)
            }
        }];);
    }

    let mut embed_option1_stream = TokenStream::new();
    if !embed_option1_fields.is_empty() {
        quote_into::quote_into!(embed_option1_stream += [#{
            for (namei,name) in embed_option1_fields.iter() {
                quote_into::quote_into!(embed_option1_stream += (embed.insert(#name.to_string(), self.#namei.as_deref().unwrap())),)
            }
        }];);
    }

    let mut linear_option1_stream_assign = TokenStream::new();
    if !linear_option1_fields.is_empty() {
        quote_into::quote_into!(linear_option1_stream_assign += [#{
            for (name, n) in linear_option1_fields.iter() {
                linear_option1_stream_assign.extend(quote::quote!((self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.linear.get(#n).unwrap().clone()))),))
            }
        }];);
    }

    let mut linear_merge_option1_stream_assign = TokenStream::new();
    if !linear_option1_fields.is_empty() {
        quote_into::quote_into!(linear_merge_option1_stream_assign += [#{
            for (name, n) in linear_option1_fields.iter() {
                linear_merge_option1_stream_assign.extend(quote::quote!(({
                    (new_layers.linear.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for option linear.");
                    self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.linear.get(#n).unwrap().clone()))
                }),))
            }
        }];);
    }

    let mut conv1d_option1_stream_assign = TokenStream::new();
    if !conv1d_option1_fields.is_empty() {
        quote_into::quote_into!(conv1d_option1_stream_assign += [#{
            for (name, n) in conv1d_option1_fields.iter() {
                conv1d_option1_stream_assign.extend(quote::quote!((self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.conv1d.get(#n).unwrap().clone()))),))
            }
        }];);
    }

    let mut conv1d_merge_option1_stream_assign = TokenStream::new();
    if !conv1d_option1_fields.is_empty() {
        quote_into::quote_into!(conv1d_merge_option1_stream_assign += [#{
            for (name, n) in conv1d_option1_fields.iter() {
                conv1d_merge_option1_stream_assign.extend(quote::quote!(({
                    (new_layers.conv1d.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for option conv1d.");
                    self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.conv1d.get(#n).unwrap().clone()))
                }),))
            }
        }];);
    }

    let mut conv2d_option1_stream_assign = TokenStream::new();
    if !conv2d_option1_fields.is_empty() {
        quote_into::quote_into!(conv2d_option1_stream_assign += [#{
            for (name, n) in conv2d_option1_fields.iter() {
                conv2d_option1_stream_assign.extend(quote::quote!((self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.conv2d.get(#n).unwrap().clone()))),))
            }
        }];);
    }

    let mut conv2d_merge_option1_stream_assign = TokenStream::new();
    if !conv2d_option1_fields.is_empty() {
        quote_into::quote_into!(conv2d_merge_option1_stream_assign += [#{
            for (name, n) in conv2d_option1_fields.iter() {
                conv2d_merge_option1_stream_assign.extend(quote::quote!(({
                    (new_layers.conv2d.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for option conv2d.");
                    self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.conv2d.get(#n).unwrap().clone()))
                }),))
            }
        }];);
    }

    let mut embed_option1_stream_assign = TokenStream::new();
    if !embed_option1_fields.is_empty() {
        quote_into::quote_into!(embed_option1_stream_assign += [#{
            for (name, n) in embed_option1_fields.iter() {
                embed_option1_stream_assign.extend(quote::quote!((self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.embed.get(#n).unwrap().clone()))),))
            }
        }];);
    }

    let mut embed_merge_option1_stream_assign = TokenStream::new();
    if !embed_option1_fields.is_empty() {
        quote_into::quote_into!(embed_merge_option1_stream_assign += [#{
            for (name, n) in embed_option1_fields.iter() {
                embed_merge_option1_stream_assign.extend(quote::quote!(({
                    (new_layers.embed.get_mut(#n).unwrap().clone()).merge_weights().expect("Merge failed for option embed.");
                    self.#name = ::std::option::Option::Some(::std::sync::Arc::new(new_layers.embed.get(#n).unwrap().clone()))
                }),))
            }
        }];);
    }

    let mut stream = TokenStream::new();
    quote_into::quote_into! { stream +=
        impl #st_name {
            /// Be sure to provide a configuration for each type!
            pub fn get_lora_model<'a>(&'a mut self, lora_config: candle_lora::LoraConfig, vb: &candle_nn::VarBuilder, linear_config: Option<candle_lora::LoraLinearConfig>, conv1d_config: Option<candle_lora::LoraConv1dConfig>, conv2d_config: Option<candle_lora::LoraConv2dConfig>, embed_config: Option<candle_lora::LoraEmbeddingConfig>) {
                let mut linear: ::std::collections::HashMap<String, &dyn candle_lora::LinearLayerLike> = ::std::collections::HashMap::new();
                let mut conv1d: ::std::collections::HashMap<String, &dyn candle_lora::Conv1dLayerLike> = ::std::collections::HashMap::new();
                let mut conv2d: ::std::collections::HashMap<String, &dyn candle_lora::Conv2dLayerLike> = ::std::collections::HashMap::new();
                let mut embed: ::std::collections::HashMap<String, &dyn candle_lora::EmbeddingLayerLike> = ::std::collections::HashMap::new();

                #linear_stream
                #conv1d_stream
                #conv2d_stream
                #embed_stream

                #linear_option1_stream
                #conv1d_option1_stream
                #conv2d_option1_stream
                #embed_option1_stream

                if !linear.is_empty() && linear_config.is_none() {
                    panic!("Config not specified for linear layers.");
                }
                if !conv1d.is_empty() && conv1d_config.is_none() {
                    panic!("Config not specified for conv1d layers.");
                }
                if !conv2d.is_empty() && conv2d_config.is_none() {
                    panic!("Config not specified for conv2d layers.");
                }
                if !embed.is_empty() && embed_config.is_none() {
                    panic!("Config not specified for embedding layers.");
                }

                let mut builder = candle_lora::SelectedLayersBuilder::new();
                if linear_config.is_some() {
                    builder = builder.add_linear_layers(linear, linear_config.unwrap());
                }
                if conv1d_config.is_some() {
                    builder = builder.add_conv1d_layers(conv1d, conv1d_config.unwrap());
                }
                if conv2d_config.is_some() {
                    builder = builder.add_conv2d_layers(conv2d, conv2d_config.unwrap());
                }
                if embed_config.is_some() {
                    builder = builder.add_embed_layers(embed, embed_config.unwrap());
                }
                let selection = builder.build();

                let new_layers = candle_lora::Lora::convert_model(selection, lora_config, &vb);

                #linear_stream_assign
                #conv1d_stream_assign
                #conv2d_stream_assign
                #embed_stream_assign

                #linear_option1_stream_assign
                #conv1d_option1_stream_assign
                #conv2d_option1_stream_assign
                #embed_option1_stream_assign
            }

            /// Be sure to provide a configuration for each type!
            pub fn get_merged_lora_model<'a>(&'a mut self, lora_config: candle_lora::LoraConfig, vb: &candle_nn::VarBuilder, linear_config: Option<candle_lora::LoraLinearConfig>, conv1d_config: Option<candle_lora::LoraConv1dConfig>, conv2d_config: Option<candle_lora::LoraConv2dConfig>, embed_config: Option<candle_lora::LoraEmbeddingConfig>) {
                use candle_lora::Merge;
                let mut linear: ::std::collections::HashMap<String, &dyn candle_lora::LinearLayerLike> = ::std::collections::HashMap::new();
                let mut conv1d: ::std::collections::HashMap<String, &dyn candle_lora::Conv1dLayerLike> = ::std::collections::HashMap::new();
                let mut conv2d: ::std::collections::HashMap<String, &dyn candle_lora::Conv2dLayerLike> = ::std::collections::HashMap::new();
                let mut embed: ::std::collections::HashMap<String, &dyn candle_lora::EmbeddingLayerLike> = ::std::collections::HashMap::new();

                #linear_stream
                #conv1d_stream
                #conv2d_stream
                #embed_stream

                #linear_option1_stream
                #conv1d_option1_stream
                #conv2d_option1_stream
                #embed_option1_stream

                if !linear.is_empty() && linear_config.is_none() {
                    panic!("Config not specified for linear layers.");
                }
                if !conv1d.is_empty() && conv1d_config.is_none() {
                    panic!("Config not specified for conv1d layers.");
                }
                if !conv2d.is_empty() && conv2d_config.is_none() {
                    panic!("Config not specified for conv2d layers.");
                }
                if !embed.is_empty() && embed_config.is_none() {
                    panic!("Config not specified for embedding layers.");
                }

                let mut builder = candle_lora::SelectedLayersBuilder::new();
                if linear_config.is_some() {
                    builder = builder.add_linear_layers(linear, linear_config.unwrap());
                }
                if conv1d_config.is_some() {
                    builder = builder.add_conv1d_layers(conv1d, conv1d_config.unwrap());
                }
                if conv2d_config.is_some() {
                    builder = builder.add_conv2d_layers(conv2d, conv2d_config.unwrap());
                }
                if embed_config.is_some() {
                    builder = builder.add_embed_layers(embed, embed_config.unwrap());
                }
                let selection = builder.build();

                let mut new_layers = candle_lora::Lora::convert_model(selection, lora_config, &vb);

                #linear_merge_stream_assign
                #conv1d_merge_stream_assign
                #conv2d_merge_stream_assign
                #embed_merge_stream_assign

                #linear_merge_option1_stream_assign
                #conv1d_merge_option1_stream_assign
                #conv2d_merge_option1_stream_assign
                #embed_merge_option1_stream_assign
            }

            pub fn get_tensors(&self) -> ::std::collections::HashMap<String, Tensor> {
                let mut output = ::std::collections::HashMap::new();
                #linear_get
                #conv1d_get
                #conv2d_get
                #embed_get
                output
            }
        }
    }

    stream.into()
}
